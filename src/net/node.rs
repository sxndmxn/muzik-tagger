use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    Multiaddr, PeerId, Swarm, SwarmBuilder,
    identify, kad,
    request_response::{self, ProtocolSupport},
    swarm::NetworkBehaviour,
};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::models::is_zero_vec;
use crate::recommend;

use super::protocol::{
    MzkCodec, PROTOCOL_NAME, PeerResult, SearchQuery, SearchResponse,
};

/// Combined network behaviour.
#[derive(NetworkBehaviour)]
pub struct MzkBehaviour {
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
    pub search: request_response::Behaviour<MzkCodec>,
}

/// Events from the node to the main loop.
pub enum NodeEvent {
    PeerConnected(PeerId),
    PeerDisconnected(PeerId),
    SearchRequest {
        peer: PeerId,
        channel: request_response::ResponseChannel<SearchResponse>,
        query: SearchQuery,
    },
    SearchResponse {
        peer: PeerId,
        response: SearchResponse,
    },
}

/// Build and start a libp2p swarm.
pub fn build_swarm() -> Result<Swarm<MzkBehaviour>> {
    let swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_quic()
        .with_behaviour(|key| {
            let peer_id = key.public().to_peer_id();

            // Kademlia
            let store = kad::store::MemoryStore::new(peer_id);
            let kademlia = kad::Behaviour::with_config(
                peer_id,
                store,
                kad::Config::new(libp2p::StreamProtocol::new("/mzk/kad/1.0.0")),
            );

            // Identify
            let identify = identify::Behaviour::new(identify::Config::new(
                "/mzk/id/1.0.0".to_string(),
                key.public(),
            ));

            // Request-response for search
            let search = request_response::Behaviour::new(
                [(PROTOCOL_NAME, ProtocolSupport::Full)],
                request_response::Config::default()
                    .with_request_timeout(Duration::from_secs(30)),
            );

            Ok(MzkBehaviour {
                kademlia,
                identify,
                search,
            })
        })?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    Ok(swarm)
}

/// Start the P2P node as a background task.
pub async fn run_node(
    port: u16,
    bootstrap: Option<String>,
    event_tx: mpsc::UnboundedSender<NodeEvent>,
) -> Result<()> {
    let mut swarm = build_swarm()?;

    let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{port}").parse()?;
    swarm.listen_on(listen_addr.clone())?;

    let local_peer_id = *swarm.local_peer_id();
    info!("Local peer ID: {local_peer_id}");
    info!("Listening on {listen_addr}");

    // Bootstrap if a peer address is provided
    if let Some(addr_str) = bootstrap {
        let addr: Multiaddr = addr_str
            .parse()
            .context("invalid bootstrap multiaddr")?;

        // Extract peer ID from the multiaddr if present
        if let Some(libp2p::multiaddr::Protocol::P2p(peer_id)) = addr.iter().last() {
            swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer_id, addr.clone());
            swarm.dial(addr)?;
            info!("Dialing bootstrap peer {peer_id}");
        } else {
            warn!("Bootstrap address missing /p2p/<peer_id> suffix");
        }
    }

    // Start Kademlia bootstrap
    let _ = swarm.behaviour_mut().kademlia.bootstrap();

    while let Some(event) = swarm.next().await {
        match event {
            libp2p::swarm::SwarmEvent::Behaviour(MzkBehaviourEvent::Search(
                request_response::Event::Message { peer, message, .. },
            )) => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    let _ = event_tx.send(NodeEvent::SearchRequest {
                        peer,
                        channel,
                        query: request,
                    });
                }
                request_response::Message::Response { response, .. } => {
                    let _ = event_tx.send(NodeEvent::SearchResponse {
                        peer,
                        response,
                    });
                }
            },
            libp2p::swarm::SwarmEvent::ConnectionEstablished {
                peer_id,
                connection_id: _,
                endpoint: _,
                num_established: _,
                concurrent_dial_errors: _,
                established_in: _,
            } => {
                let _ = event_tx.send(NodeEvent::PeerConnected(peer_id));
            }
            libp2p::swarm::SwarmEvent::ConnectionClosed {
                peer_id,
                connection_id: _,
                endpoint: _,
                num_established: _,
                cause: _,
            } => {
                let _ = event_tx.send(NodeEvent::PeerDisconnected(peer_id));
            }
            libp2p::swarm::SwarmEvent::Behaviour(MzkBehaviourEvent::Identify(
                identify::Event::Received { peer_id, info, .. },
            )) => {
                for addr in info.listen_addrs {
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr);
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Handle an incoming search query locally and return results.
pub fn handle_search_query(query: &SearchQuery) -> SearchResponse {
    let Ok(db) = crate::db::open_db() else {
        return SearchResponse { results: vec![] };
    };

    let Ok(tracks) = crate::db::all_tracks(&db) else {
        return SearchResponse { results: vec![] };
    };

    let local_peer_id = "local".to_string();

    match query {
        SearchQuery::TextSearch {
            query_vec,
            max_results,
        } => {
            let mut scored: Vec<PeerResult> = tracks
                .iter()
                .filter(|t| !is_zero_vec(&t.audio_vec))
                .map(|t| {
                    let score = recommend::cosine_sim(query_vec, &t.audio_vec);
                    PeerResult {
                        score,
                        artist: t.artist.clone(),
                        title: t.title.clone(),
                        album: t.album.clone(),
                        year: t.year,
                        peer_id: local_peer_id.clone(),
                    }
                })
                .collect();
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(*max_results);
            SearchResponse { results: scored }
        }
        SearchQuery::SimilarSearch {
            metadata_vec,
            audio_vec,
            weight_audio,
            max_results,
        } => {
            let query_meta = metadata_vec.as_deref();
            let query_audio = audio_vec.as_deref();

            let mut scored: Vec<PeerResult> = tracks
                .iter()
                .map(|t| {
                    let score = recommend::hybrid_score(
                        query_meta,
                        query_audio,
                        &t.metadata_vec,
                        &t.audio_vec,
                        *weight_audio,
                    );
                    PeerResult {
                        score,
                        artist: t.artist.clone(),
                        title: t.title.clone(),
                        album: t.album.clone(),
                        year: t.year,
                        peer_id: local_peer_id.clone(),
                    }
                })
                .collect();
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(*max_results);
            SearchResponse { results: scored }
        }
    }
}
