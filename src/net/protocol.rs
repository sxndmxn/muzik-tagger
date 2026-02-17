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
    /// Content fingerprint derived from CLAP audio embedding. Two-tier:
    /// - `fine`: 512-bit sign-quantized hash (1 bit per dimension)
    /// - `coarse`: 32-bit hash (average blocks of 16 dims, then sign-quantize)
    ///
    /// Same song in different codecs (FLAC/MP3/WAV) produces near-identical CLAP
    /// vectors. The coarse hash survives codec artifacts that flip individual fine bits.
    pub audio_fingerprint: Option<AudioFingerprint>,
}

/// Two-tier content fingerprint for codec-robust deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFingerprint {
    /// 512-bit sign-quantized hash (64 bytes hex). Per-dimension resolution.
    pub fine: String,
    /// 32-bit coarse hash (4 bytes hex). Averages blocks of 16 dims before
    /// sign-quantizing. Robust to lossy codec artifacts (MP3 vs FLAC).
    pub coarse: String,
}

/// Compute a two-tier content fingerprint from an audio embedding vector.
///
/// **Fine** (512 bits): each dimension → 1 bit (positive = 1).
/// Catches exact/near-exact duplicates.
///
/// **Coarse** (32 bits): average groups of 16 adjacent dimensions, then
/// sign-quantize. Lossy compression shifts individual dimensions but rarely
/// flips the average sign of a 16-dim block. Handles FLAC vs MP3 vs AAC.
#[must_use]
pub fn audio_fingerprint(audio_vec: &[f32]) -> AudioFingerprint {
    // Fine: 1 bit per dimension
    let mut fine_bytes = vec![0u8; audio_vec.len().div_ceil(8)];
    for (i, &val) in audio_vec.iter().enumerate() {
        if val > 0.0 {
            fine_bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }

    // Coarse: average blocks of 16, then sign-quantize
    let block_size = 16;
    let num_blocks = audio_vec.len().div_ceil(block_size);
    let mut coarse_bytes = vec![0u8; num_blocks.div_ceil(8)];
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(audio_vec.len());
        let block = &audio_vec[start..end];
        #[allow(clippy::cast_precision_loss)]
        let mean: f32 = block.iter().sum::<f32>() / block.len() as f32;
        if mean > 0.0 {
            coarse_bytes[block_idx / 8] |= 1 << (7 - (block_idx % 8));
        }
    }

    AudioFingerprint {
        fine: hex::encode(&fine_bytes),
        coarse: hex::encode(&coarse_bytes),
    }
}

/// Check if two fingerprints represent the same audio content.
///
/// Match criteria (either triggers dedup):
/// 1. Coarse hashes within 3 bits (out of 32) — survives codec differences
/// 2. Fine hashes within 48 bits (out of 512) — catches near-exact matches
#[must_use]
pub fn fingerprints_match(a: &AudioFingerprint, b: &AudioFingerprint) -> bool {
    // Coarse match: very robust, low resolution
    if hamming_distance(&a.coarse, &b.coarse).is_some_and(|d| d <= 3) {
        return true;
    }

    // Fine match: higher resolution, tighter threshold
    hamming_distance(&a.fine, &b.fine).is_some_and(|d| d <= 48)
}

/// Hamming distance between two hex-encoded bitstrings.
/// Returns `None` if lengths differ or hex is invalid.
#[must_use]
pub fn hamming_distance(a: &str, b: &str) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    let a_bytes = hex::decode(a).ok()?;
    let b_bytes = hex::decode(b).ok()?;
    Some(
        a_bytes
            .iter()
            .zip(b_bytes.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum(),
    )
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
