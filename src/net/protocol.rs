use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response;
use serde::{Deserialize, Serialize};

/// The protocol name for mzk search.
pub const PROTOCOL_NAME: libp2p::StreamProtocol =
    libp2p::StreamProtocol::new("/mzk/search/1.0.0");

/// A search query sent to peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchQuery {
    /// Natural language text search (CLAP text embedding).
    TextSearch {
        query_vec: Vec<f32>,
        max_results: usize,
    },
    /// Similarity search by track (metadata + audio vectors).
    SimilarSearch {
        metadata_vec: Option<Vec<f32>>,
        audio_vec: Option<Vec<f32>>,
        weight_audio: f32,
        max_results: usize,
    },
}

/// A single result from a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerResult {
    pub score: f32,
    pub artist: String,
    pub title: String,
    pub album: String,
    pub year: i32,
    pub peer_id: String,
}

/// Response from a peer to a search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<PeerResult>,
}

/// Codec for msgpack serialization of search messages.
#[derive(Debug, Clone, Default)]
pub struct MzkCodec;

#[async_trait]
impl request_response::Codec for MzkCodec {
    type Protocol = libp2p::StreamProtocol;
    type Request = SearchQuery;
    type Response = SearchResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        rmp_serde::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        rmp_serde::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let buf = rmp_serde::to_vec(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        io.write_all(&buf).await?;
        io.close().await?;
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let buf = rmp_serde::to_vec(&resp)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        io.write_all(&buf).await?;
        io.close().await?;
        Ok(())
    }
}
