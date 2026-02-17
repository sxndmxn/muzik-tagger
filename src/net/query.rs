use std::collections::HashSet;
use std::time::Duration;

use libp2p::PeerId;
use tokio::sync::mpsc;
use tokio::time::timeout;

use super::protocol::{PeerResult, SearchQuery, SearchResponse, fingerprints_match};

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
///
/// Deduplication uses two-tier audio fingerprints (content-based):
/// - **Coarse** (32-bit): averages blocks of 16 CLAP dims. Survives
///   codec differences (FLAC vs MP3 vs AAC) because lossy compression
///   shifts individual dimensions but rarely flips 16-dim block averages.
/// - **Fine** (512-bit): per-dimension sign. Catches near-exact dupes.
///
/// Same song in different formats → same coarse fingerprint → deduplicated.
/// No fingerprint → no dedup (conservative: show duplicates, don't lose results).
fn merge_results(mut results: Vec<PeerResult>) -> Vec<PeerResult> {
    // Sort by score descending — first occurrence (highest score) wins dedup
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<PeerResult> = Vec::with_capacity(results.len());

    for result in results {
        let is_duplicate = kept.iter().any(|existing| {
            if let (Some(fp_a), Some(fp_b)) =
                (&existing.audio_fingerprint, &result.audio_fingerprint)
            {
                return fingerprints_match(fp_a, fp_b);
            }
            false
        });

        if !is_duplicate {
            kept.push(result);
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::protocol::{AudioFingerprint, audio_fingerprint};

    fn make_result(
        score: f32,
        artist: &str,
        title: &str,
        peer: &str,
        fp: Option<AudioFingerprint>,
    ) -> PeerResult {
        PeerResult {
            score,
            artist: artist.to_string(),
            title: title.to_string(),
            album: "Album".to_string(),
            year: 2020,
            peer_id: peer.to_string(),
            audio_fingerprint: fp,
        }
    }

    #[test]
    fn test_same_song_different_tags_deduped() {
        // Identical audio → identical fingerprint, different metadata → deduped
        let fp = audio_fingerprint(&vec![1.0; 512]);
        let results = vec![
            make_result(0.9, "Aphex Twin", "Xtal", "peer1", Some(fp.clone())),
            make_result(0.8, "AFX", "Xtal (Original)", "peer2", Some(fp)),
            make_result(0.7, "Boards of Canada", "Roygbiv", "peer1", None),
        ];

        let merged = merge_results(results);
        assert_eq!(merged.len(), 2);
        assert!((merged[0].score - 0.9).abs() < f32::EPSILON);
        assert_eq!(merged[0].artist, "Aphex Twin");
    }

    #[test]
    fn test_different_songs_kept() {
        let fp_a = audio_fingerprint(&vec![1.0; 512]);
        let fp_b = audio_fingerprint(&vec![-1.0; 512]);
        let results = vec![
            make_result(0.9, "Artist A", "Track 1", "peer1", Some(fp_a)),
            make_result(0.8, "Artist A", "Track 1", "peer2", Some(fp_b)),
        ];

        // Completely different fingerprints = different content, even if same title
        let merged = merge_results(results);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_codec_difference_deduped_via_coarse() {
        // Same song, but MP3 encoding flips some individual dimensions.
        // Coarse hash (16-dim block averages) should survive this.
        let vec_a = vec![1.0f32; 512];
        let mut vec_b = vec_a.clone();
        // Flip 1 dim per 16-dim block — coarse hash unaffected because
        // the block average stays positive (15 positive + 1 negative > 0)
        for i in (0..512).step_by(16) {
            vec_b[i] = -0.5;
        }
        let fp_a = audio_fingerprint(&vec_a);
        let fp_b = audio_fingerprint(&vec_b);

        let results = vec![
            make_result(0.9, "Artist", "Track (FLAC)", "peer1", Some(fp_a)),
            make_result(0.8, "Artist", "Track (MP3)", "peer2", Some(fp_b)),
        ];

        let merged = merge_results(results);
        assert_eq!(merged.len(), 1, "codec variants should dedup via coarse hash");
    }

    #[test]
    fn test_fine_threshold_dedup() {
        // Small perturbation — fine hash within 48-bit threshold
        let vec_a = vec![1.0f32; 512];
        let mut vec_b = vec_a.clone();
        for v in vec_b.iter_mut().take(20) {
            *v = -1.0;
        }
        let fp_a = audio_fingerprint(&vec_a);
        let fp_b = audio_fingerprint(&vec_b);

        let results = vec![
            make_result(0.9, "A", "T1", "p1", Some(fp_a)),
            make_result(0.8, "A", "T1", "p2", Some(fp_b)),
        ];

        let merged = merge_results(results);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_sorts_by_score() {
        let results = vec![
            make_result(0.5, "Low", "T1", "p1", None),
            make_result(0.9, "High", "T2", "p2", None),
        ];

        let merged = merge_results(results);
        assert!((merged[0].score - 0.9).abs() < f32::EPSILON);
        assert_eq!(merged[0].artist, "High");
    }

    #[test]
    fn test_no_fingerprints_keeps_all() {
        // No fingerprints → no dedup (conservative)
        let results = vec![
            make_result(0.9, "Aphex Twin", "Xtal", "peer1", None),
            make_result(0.8, "AFX", "Xtal (Original)", "peer2", None),
        ];

        let merged = merge_results(results);
        assert_eq!(merged.len(), 2);
    }
}
