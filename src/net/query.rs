use std::collections::HashSet;
use std::time::Duration;

use libp2p::PeerId;
use tokio::sync::mpsc;
use tokio::time::timeout;

use super::protocol::{PeerResult, SearchQuery, SearchResponse};

/// Fan out a query to connected peers and aggregate results.
pub async fn federated_query(
    query: SearchQuery,
    peers: Vec<PeerId>,
    tx: &mpsc::UnboundedSender<(PeerId, SearchQuery)>,
    rx: &mut mpsc::UnboundedReceiver<(PeerId, SearchResponse)>,
    timeout_secs: u64,
) -> Vec<PeerResult> {
    if peers.is_empty() {
        return vec![];
    }

    // Send query to all peers
    for peer in &peers {
        let _ = tx.send((*peer, query.clone()));
    }

    // Collect responses with timeout
    let mut all_results = Vec::new();
    let mut responded = HashSet::new();
    let deadline = Duration::from_secs(timeout_secs);

    while responded.len() < peers.len() {
        match timeout(deadline, rx.recv()).await {
            Ok(Some((peer, response))) => {
                responded.insert(peer);
                all_results.extend(response.results);
            }
            Ok(None) | Err(_) => break,
        }
    }

    // Merge, deduplicate, and re-rank
    merge_results(all_results)
}

/// Merge results from multiple peers, deduplicate, and re-rank by score.
fn merge_results(mut results: Vec<PeerResult>) -> Vec<PeerResult> {
    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate by (artist, title, album) — keep highest score
    let mut seen = HashSet::new();
    results.retain(|r| {
        let key = format!(
            "{}||{}||{}",
            r.artist.to_lowercase(),
            r.title.to_lowercase(),
            r.album.to_lowercase(),
        );
        seen.insert(key)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_deduplicates() {
        let results = vec![
            PeerResult {
                score: 0.9,
                artist: "Artist A".to_string(),
                title: "Track 1".to_string(),
                album: "Album X".to_string(),
                year: 2020,
                peer_id: "peer1".to_string(),
            },
            PeerResult {
                score: 0.8,
                artist: "Artist A".to_string(),
                title: "Track 1".to_string(),
                album: "Album X".to_string(),
                year: 2020,
                peer_id: "peer2".to_string(),
            },
            PeerResult {
                score: 0.7,
                artist: "Artist B".to_string(),
                title: "Track 2".to_string(),
                album: "Album Y".to_string(),
                year: 2021,
                peer_id: "peer1".to_string(),
            },
        ];

        let merged = merge_results(results);
        assert_eq!(merged.len(), 2);
        assert!((merged[0].score - 0.9).abs() < f32::EPSILON);
        assert_eq!(merged[1].artist, "Artist B");
    }

    #[test]
    fn test_merge_sorts_by_score() {
        let results = vec![
            PeerResult {
                score: 0.5,
                artist: "Low".to_string(),
                title: "T1".to_string(),
                album: "A1".to_string(),
                year: 2020,
                peer_id: "p1".to_string(),
            },
            PeerResult {
                score: 0.9,
                artist: "High".to_string(),
                title: "T2".to_string(),
                album: "A2".to_string(),
                year: 2021,
                peer_id: "p2".to_string(),
            },
        ];

        let merged = merge_results(results);
        assert!((merged[0].score - 0.9).abs() < f32::EPSILON);
        assert_eq!(merged[0].artist, "High");
    }
}
