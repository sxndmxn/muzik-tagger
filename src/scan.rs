use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use lofty::prelude::*;
use lofty::probe::Probe;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::db;
use crate::models::{AUDIO_EXTENSIONS, Album, Track};

fn track_id(filepath: &str) -> String {
    let hash = Sha256::digest(filepath.as_bytes());
    hex::encode(&hash[..8])
}

fn album_id(artist: &str, album: &str) -> String {
    let key = format!("{}||{}", artist.to_lowercase().trim(), album.to_lowercase().trim());
    let hash = Sha256::digest(key.as_bytes());
    hex::encode(&hash[..8])
}

fn get_tag_string(tag: &dyn Accessor, getter: fn(&dyn Accessor) -> Option<std::borrow::Cow<'_, str>>) -> String {
    getter(tag).map_or_else(String::new, |v| v.to_string())
}

fn extract_metadata(filepath: &Path) -> Option<Track> {
    let tagged_file = Probe::open(filepath)
        .ok()?
        .read()
        .ok()?;

    let tag = tagged_file.primary_tag().or_else(|| tagged_file.first_tag())?;

    let artist = get_tag_string(tag, |t| t.artist());
    let album = get_tag_string(tag, |t| t.album());
    let title = get_tag_string(tag, |t| t.title());
    let genre = get_tag_string(tag, |t| t.genre());
    let year = tag.year().map_or(0, u32::cast_signed);
    let track_num = tag.track().map_or(0, u32::cast_signed);

    // Get album_artist from ItemKey
    let album_artist = tag
        .get_string(&lofty::tag::ItemKey::AlbumArtist)
        .unwrap_or(&artist)
        .to_string();

    let filepath_str = filepath.to_string_lossy().to_string();

    // Fallback: infer from directory structure
    let final_artist = if artist.is_empty() {
        filepath
            .parent()
            .and_then(|p| p.parent())
            .map_or_else(String::new, |p| p.file_name().map_or_else(String::new, |n| n.to_string_lossy().to_string()))
    } else {
        artist
    };
    let final_album = if album.is_empty() {
        filepath
            .parent()
            .map_or_else(String::new, |p| p.file_name().map_or_else(String::new, |n| n.to_string_lossy().to_string()))
    } else {
        album
    };
    let final_title = if title.is_empty() {
        filepath
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().to_string())
    } else {
        title
    };

    Some(Track {
        id: track_id(&filepath_str),
        title: final_title,
        artist: final_artist,
        album_artist: if album_artist.is_empty() {
            "Unknown".to_string()
        } else {
            album_artist
        },
        album: final_album,
        genre,
        year,
        track_num,
        filepath: filepath_str,
        metadata_vec: vec![0.0; crate::models::METADATA_DIM],
        audio_vec: vec![0.0; crate::models::AUDIO_DIM],
    })
}

/// Scan a music library directory and store metadata in `LanceDB`.
///
/// Returns `(track_count, album_count)`.
pub fn scan_library(library_path: &Path, quiet: bool) -> Result<(usize, usize)> {
    if !library_path.is_dir() {
        bail!("{} is not a directory", library_path.display());
    }

    // Collect audio files
    let audio_files: Vec<_> = WalkDir::new(library_path)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        })
        .map(walkdir::DirEntry::into_path)
        .collect();

    if audio_files.is_empty() {
        bail!("No audio files found in {}", library_path.display());
    }

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(audio_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg}")
                .context("progress bar template")?,
        );
        pb.set_message("Scanning tracks");
        pb
    };

    let mut db = db::open_db()?;
    let mut all_tracks: Vec<Track> = Vec::with_capacity(audio_files.len());
    let mut album_data: HashMap<String, AlbumAccumulator> = HashMap::new();

    for filepath in &audio_files {
        pb.inc(1);
        let Some(track) = extract_metadata(filepath) else {
            continue;
        };

        // Accumulate album info
        let aid = album_id(&track.artist, &track.album);
        let entry = album_data.entry(aid.clone()).or_insert_with(|| AlbumAccumulator {
            id: aid,
            artist: track.album_artist.clone(),
            album: track.album.clone(),
            year: track.year,
            genres: HashMap::new(),
            track_count: 0,
        });
        entry.track_count += 1;
        if !track.genre.is_empty() {
            *entry.genres.entry(track.genre.clone()).or_insert(0) += 1;
        }

        all_tracks.push(track);
    }

    let total_tracks = all_tracks.len();
    db::write_all_tracks(&mut db, &all_tracks)?;

    // Build and write albums
    let albums: Vec<Album> = album_data
        .into_values()
        .map(|acc| {
            let mut sorted_genres: Vec<_> = acc.genres.into_iter().collect();
            sorted_genres.sort_by(|a, b| b.1.cmp(&a.1));
            let genres_str = sorted_genres
                .into_iter()
                .map(|(g, _)| g)
                .collect::<Vec<_>>()
                .join(", ");
            Album {
                id: acc.id,
                artist: acc.artist,
                album: acc.album,
                year: acc.year,
                genres: genres_str,
                track_count: acc.track_count,
                metadata_vec: vec![0.0; crate::models::METADATA_DIM],
                audio_vec: vec![0.0; crate::models::AUDIO_DIM],
            }
        })
        .collect();

    let album_count = albums.len();
    db::write_all_albums(&mut db, &albums)?;

    pb.finish_and_clear();
    Ok((total_tracks, album_count))
}

struct AlbumAccumulator {
    id: String,
    artist: String,
    album: String,
    year: i32,
    genres: HashMap<String, i32>,
    track_count: i32,
}
