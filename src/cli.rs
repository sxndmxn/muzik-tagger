use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "mzk", about = "Distributed music discovery network")]
pub struct Cli {
    /// Suppress stderr output (progress bars, status messages).
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Output results as JSON lines (NDJSON).
    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Scan a music library and extract metadata from audio tags.
    Scan {
        /// Path to music library directory.
        path: PathBuf,
    },

    /// Generate vector embeddings for tracks.
    Embed {
        /// Generate metadata text embeddings.
        #[arg(long)]
        metadata: bool,

        /// Generate CLAP audio embeddings.
        #[arg(long)]
        audio: bool,

        /// Generate both metadata and audio embeddings.
        #[arg(long)]
        all: bool,
    },

    /// Find similar tracks.
    Similar {
        /// Artist name.
        artist: String,

        /// Track title (or album name with --album).
        title: String,

        /// Search for similar albums instead of tracks.
        #[arg(long)]
        album: bool,

        /// Number of results.
        #[arg(short, default_value = "10")]
        n: usize,

        /// Audio embedding weight (0.0-1.0).
        #[arg(short = 'w', long = "weight-audio", default_value = "0.7")]
        weight_audio: f32,
    },

    /// Search your library by natural language description.
    Search {
        /// Description of what you want to hear.
        query: String,

        /// Number of results.
        #[arg(short, default_value = "10")]
        n: usize,
    },

    /// Generate a radio playlist seeded from a track.
    Radio {
        /// Seed artist.
        artist: String,

        /// Seed track title.
        title: String,

        /// Playlist length.
        #[arg(short, default_value = "20")]
        n: usize,

        /// Audio embedding weight (0.0-1.0).
        #[arg(short = 'w', long = "weight-audio", default_value = "0.7")]
        weight_audio: f32,
    },

    /// Show a taste profile summary of your library.
    Profile,

    /// Start P2P node.
    Serve {
        /// TCP listen port.
        #[arg(long, default_value = "4001")]
        port: u16,

        /// Bootstrap peer multiaddr (e.g. `/ip4/1.2.3.4/tcp/4001/p2p/<peer_id>`).
        #[arg(long)]
        bootstrap: Option<String>,
    },

    /// List connected peers.
    Peers,

    /// Federated network commands.
    Network {
        #[command(subcommand)]
        command: NetworkCommand,
    },
}

#[derive(Subcommand)]
pub enum NetworkCommand {
    /// Federated natural language search across the network.
    Search {
        /// Description of what you want to hear.
        query: String,

        /// Number of results.
        #[arg(short, default_value = "10")]
        n: usize,
    },

    /// Federated similarity search across the network.
    Similar {
        /// Artist name.
        artist: String,

        /// Track title.
        title: String,

        /// Number of results.
        #[arg(short, default_value = "10")]
        n: usize,

        /// Audio embedding weight (0.0-1.0).
        #[arg(short = 'w', long = "weight-audio", default_value = "0.7")]
        weight_audio: f32,
    },
}
