#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

pub const METADATA_DIM: usize = 384;
pub const AUDIO_DIM: usize = 512;

pub const AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "flac", "ogg", "opus", "m4a", "wav", "aiff", "wma",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_field_names)]
pub struct Track {
    pub id: String,
    pub title: String,
    pub artist: String,
    pub album_artist: String,
    pub album: String,
    pub genre: String,
    pub year: i32,
    pub track_num: i32,
    pub filepath: String,
    #[serde(default = "default_metadata_vec")]
    pub metadata_vec: Vec<f32>,
    #[serde(default = "default_audio_vec")]
    pub audio_vec: Vec<f32>,
}

impl Track {
    /// Text string used for metadata embedding.
    #[must_use]
    pub fn metadata_text(&self) -> String {
        let mut parts = vec![
            self.artist.as_str(),
            self.title.as_str(),
            self.album.as_str(),
            self.genre.as_str(),
        ];
        let year_str = self.year.to_string();
        if self.year > 0 {
            parts.push(&year_str);
        }
        parts
            .into_iter()
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_field_names)]
pub struct Album {
    pub id: String,
    pub artist: String,
    pub album: String,
    pub year: i32,
    pub genres: String,
    pub track_count: i32,
    #[serde(default = "default_metadata_vec")]
    pub metadata_vec: Vec<f32>,
    #[serde(default = "default_audio_vec")]
    pub audio_vec: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub score: f32,
    pub artist: String,
    pub title: String,
    pub album: String,
    pub year: i32,
    pub filepath: String,
}

impl SearchResult {
    /// Format as TSV line: score, artist, title, album, year, filepath.
    #[must_use]
    pub fn to_tsv(&self) -> String {
        format!(
            "{:.4}\t{}\t{}\t{}\t{}\t{}",
            self.score, self.artist, self.title, self.album, self.year, self.filepath,
        )
    }
}

fn default_metadata_vec() -> Vec<f32> {
    vec![0.0; METADATA_DIM]
}

fn default_audio_vec() -> Vec<f32> {
    vec![0.0; AUDIO_DIM]
}

#[must_use]
pub fn is_zero_vec(vec: &[f32]) -> bool {
    vec.is_empty() || vec.iter().take(8).all(|&v| v == 0.0)
}
