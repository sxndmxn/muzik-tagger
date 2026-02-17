use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::db;
use crate::models::{SearchResult, is_zero_vec};

/// Default weight for audio embeddings in hybrid scoring (70% audio, 30% metadata).
#[allow(dead_code)]
pub const DEFAULT_WEIGHT_AUDIO: f32 = 0.7;

/// Cosine similarity between two vectors.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute hybrid similarity score from metadata and audio vectors.
pub fn hybrid_score(
    query_meta: Option<&[f32]>,
    query_audio: Option<&[f32]>,
    cand_meta: &[f32],
    cand_audio: &[f32],
    weight_audio: f32,
) -> f32 {
    let meta_score = match query_meta {
        Some(qm) if !is_zero_vec(cand_meta) => cosine_sim(qm, cand_meta),
        _ => 0.0,
    };
    let has_meta = query_meta.is_some() && !is_zero_vec(cand_meta);

    let audio_score = match query_audio {
        Some(qa) if !is_zero_vec(cand_audio) => cosine_sim(qa, cand_audio),
        _ => 0.0,
    };
    let has_audio = query_audio.is_some() && !is_zero_vec(cand_audio);

    match (has_audio, has_meta) {
        (true, true) => (1.0 - weight_audio) * meta_score + weight_audio * audio_score,
        (true, false) => audio_score,
        (false, true) => meta_score,
        (false, false) => 0.0,
    }
}

/// Find tracks most similar to the given track.
pub fn similar_tracks(
    artist: &str,
    title: &str,
    n: usize,
    weight_audio: f32,
) -> Result<Vec<SearchResult>> {
    let db = db::open_db()?;
    let query = db::find_track(&db, artist, title)?;
    let Some(query) = query else {
        bail!("Track not found: {artist} - {title}");
    };

    let query_meta = if is_zero_vec(&query.metadata_vec) {
        None
    } else {
        Some(query.metadata_vec.as_slice())
    };
    let query_audio = if is_zero_vec(&query.audio_vec) {
        None
    } else {
        Some(query.audio_vec.as_slice())
    };

    if query_meta.is_none() && query_audio.is_none() {
        bail!("Track has no embeddings. Run 'mzk embed' first.");
    }

    let tracks = db::all_tracks(&db)?;
    let mut scored: Vec<SearchResult> = tracks
        .iter()
        .filter(|t| t.id != query.id)
        .map(|t| {
            let score = hybrid_score(
                query_meta,
                query_audio,
                &t.metadata_vec,
                &t.audio_vec,
                weight_audio,
            );
            SearchResult {
                score,
                artist: t.artist.clone(),
                title: t.title.clone(),
                album: t.album.clone(),
                year: t.year,
                filepath: t.filepath.clone(),
            }
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    Ok(scored)
}

/// Find albums most similar to the given album.
pub fn similar_albums(
    artist: &str,
    album: &str,
    n: usize,
    weight_audio: f32,
) -> Result<Vec<SearchResult>> {
    let db = db::open_db()?;
    let query = db::find_album(&db, artist, album)?;
    let Some(query) = query else {
        bail!("Album not found: {artist} - {album}");
    };

    let query_meta = if is_zero_vec(&query.metadata_vec) {
        None
    } else {
        Some(query.metadata_vec.as_slice())
    };
    let query_audio = if is_zero_vec(&query.audio_vec) {
        None
    } else {
        Some(query.audio_vec.as_slice())
    };

    if query_meta.is_none() && query_audio.is_none() {
        bail!("Album has no embeddings. Run 'mzk embed' first.");
    }

    let albums = db::all_albums(&db)?;
    let mut scored: Vec<SearchResult> = albums
        .iter()
        .filter(|a| a.id != query.id)
        .map(|a| {
            let score = hybrid_score(
                query_meta,
                query_audio,
                &a.metadata_vec,
                &a.audio_vec,
                weight_audio,
            );
            SearchResult {
                score,
                artist: a.artist.clone(),
                title: String::new(),
                album: a.album.clone(),
                year: a.year,
                filepath: String::new(),
            }
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    Ok(scored)
}

/// Search library using a natural language description via CLAP text embeddings.
pub fn search_by_text(query: &str, n: usize) -> Result<Vec<SearchResult>> {
    let query_vec = crate::embed::audio::clap_text_embed(query)?;

    let db = db::open_db()?;
    let tracks = db::all_tracks(&db)?;

    let mut scored: Vec<SearchResult> = tracks
        .iter()
        .filter(|t| !is_zero_vec(&t.audio_vec))
        .map(|t| {
            let score = cosine_sim(&query_vec, &t.audio_vec);
            SearchResult {
                score,
                artist: t.artist.clone(),
                title: t.title.clone(),
                album: t.album.clone(),
                year: t.year,
                filepath: t.filepath.clone(),
            }
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    Ok(scored)
}

/// Generate a taste profile summary from the library.
pub fn taste_profile() -> Result<TasteProfile> {
    let db = db::open_db()?;
    let tracks = db::all_tracks(&db)?;

    let mut genre_counts: HashMap<String, usize> = HashMap::new();
    let mut artist_counts: HashMap<String, usize> = HashMap::new();
    let mut decade_counts: HashMap<String, usize> = HashMap::new();

    for track in &tracks {
        if !track.genre.is_empty() {
            *genre_counts.entry(track.genre.clone()).or_default() += 1;
        }
        if !track.artist.is_empty() {
            *artist_counts.entry(track.artist.clone()).or_default() += 1;
        }
        if track.year > 0 {
            let decade = format!("{}s", (track.year / 10) * 10);
            *decade_counts.entry(decade).or_default() += 1;
        }
    }

    let mut top_genres: Vec<_> = genre_counts.into_iter().collect();
    top_genres.sort_by(|a, b| b.1.cmp(&a.1));
    top_genres.truncate(15);

    let mut top_artists: Vec<_> = artist_counts.into_iter().collect();
    top_artists.sort_by(|a, b| b.1.cmp(&a.1));
    top_artists.truncate(15);

    let mut decades: Vec<_> = decade_counts.into_iter().collect();
    decades.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(TasteProfile {
        total_tracks: tracks.len(),
        top_genres,
        top_artists,
        decades,
    })
}

/// Generate a radio-style playlist seeded from a track.
pub fn radio_playlist(
    artist: &str,
    title: &str,
    n: usize,
    weight_audio: f32,
) -> Result<Vec<SearchResult>> {
    let initial = similar_tracks(artist, title, n * 2, weight_audio)?;

    if initial.len() <= n {
        return Ok(initial);
    }

    // Pick top half directly, then diversify
    let half = n / 2;
    let mut playlist: Vec<SearchResult> = initial[..half].to_vec();
    let mut seen_artists: HashSet<String> = playlist.iter().map(|t| t.artist.clone()).collect();

    let remaining = &initial[half..];

    // Sort remaining by diversity (unseen artists first), then score
    let mut rest: Vec<&SearchResult> = remaining.iter().collect();
    rest.sort_by(|a, b| {
        let a_seen = seen_artists.contains(&a.artist);
        let b_seen = seen_artists.contains(&b.artist);
        a_seen
            .cmp(&b_seen)
            .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });

    for r in rest {
        if playlist.len() >= n {
            break;
        }
        playlist.push(r.clone());
        seen_artists.insert(r.artist.clone());
    }

    Ok(playlist)
}

#[derive(Debug, serde::Serialize)]
pub struct TasteProfile {
    pub total_tracks: usize,
    pub top_genres: Vec<(String, usize)>,
    pub top_artists: Vec<(String, usize)>,
    pub decades: Vec<(String, usize)>,
}
