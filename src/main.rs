#![deny(clippy::all, clippy::pedantic)]
#![deny(warnings)]
#![allow(clippy::module_name_repetitions)]

mod cli;
mod db;
mod embed;
mod models;
mod net;
mod recommend;
mod scan;

use std::process;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command, NetworkCommand};
use models::SearchResult;

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy)]
struct EmbedOptions {
    metadata: bool,
    audio: bool,
    all: bool,
    quiet: bool,
}

fn main() {
    let cli = Cli::parse();

    if !cli.quiet {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
            )
            .with_writer(std::io::stderr)
            .init();
    }

    let result = run(cli);

    match result {
        Ok(code) => process::exit(code),
        Err(e) => {
            eprintln!("Error: {e:#}");
            process::exit(1);
        }
    }
}

fn run(cli: Cli) -> Result<i32> {
    match cli.command {
        Command::Scan { path } => cmd_scan(&path, cli.quiet),
        Command::Embed {
            metadata,
            audio,
            all,
        } => cmd_embed(EmbedOptions {
            metadata,
            audio,
            all,
            quiet: cli.quiet,
        }),
        Command::Similar {
            artist,
            title,
            album,
            n,
            weight_audio,
        } => {
            if album {
                cmd_similar_albums(&artist, &title, n, weight_audio, cli.json)
            } else {
                cmd_similar_tracks(&artist, &title, n, weight_audio, cli.json)
            }
        }
        Command::Search { query, n } => cmd_search(&query, n, cli.json),
        Command::Radio {
            artist,
            title,
            n,
            weight_audio,
        } => cmd_radio(&artist, &title, n, weight_audio, cli.json),
        Command::Profile => cmd_profile(cli.json),
        Command::Serve { port, bootstrap } => cmd_serve(port, bootstrap),
        Command::Peers => Ok(cmd_peers()),
        Command::Network { command } => match command {
            NetworkCommand::Search { query, n } => Ok(cmd_network_search(&query, n, cli.json)),
            NetworkCommand::Similar {
                artist,
                title,
                n,
                weight_audio,
            } => Ok(cmd_network_similar(&artist, &title, n, weight_audio, cli.json)),
        },
    }
}

fn cmd_scan(path: &std::path::Path, quiet: bool) -> Result<i32> {
    if !quiet {
        eprintln!("Scanning {}...", path.display());
    }
    let (tracks, albums) = scan::scan_library(path, quiet)?;
    if !quiet {
        eprintln!("Done: {tracks} tracks, {albums} albums");
    }
    Ok(0)
}

fn cmd_embed(opts: EmbedOptions) -> Result<i32> {
    if !opts.metadata && !opts.audio && !opts.all {
        eprintln!("Specify --metadata, --audio, or --all");
        return Ok(1);
    }

    if opts.metadata || opts.all {
        if !opts.quiet {
            eprintln!("Generating metadata embeddings...");
        }
        let count = embed::metadata::embed_metadata(opts.quiet)?;
        if !opts.quiet {
            eprintln!("Embedded metadata for {count} tracks");
        }
    }

    if opts.audio || opts.all {
        if !opts.quiet {
            eprintln!("Generating CLAP audio embeddings...");
        }
        let count = embed::audio::embed_audio(opts.quiet)?;
        if count == 0 && !opts.quiet {
            eprintln!("All tracks already have audio embeddings");
        } else if !opts.quiet {
            eprintln!("Embedded audio for {count} tracks");
        }
    }

    Ok(0)
}

fn cmd_similar_tracks(
    artist: &str,
    title: &str,
    n: usize,
    weight_audio: f32,
    json: bool,
) -> Result<i32> {
    let results = recommend::similar_tracks(artist, title, n, weight_audio)?;
    if results.is_empty() {
        return Ok(2);
    }
    output_results(&results, json);
    Ok(0)
}

fn cmd_similar_albums(
    artist: &str,
    album: &str,
    n: usize,
    weight_audio: f32,
    json: bool,
) -> Result<i32> {
    let results = recommend::similar_albums(artist, album, n, weight_audio)?;
    if results.is_empty() {
        return Ok(2);
    }
    output_results(&results, json);
    Ok(0)
}

fn cmd_search(query: &str, n: usize, json: bool) -> Result<i32> {
    let results = recommend::search_by_text(query, n)?;
    if results.is_empty() {
        return Ok(2);
    }
    output_results(&results, json);
    Ok(0)
}

fn cmd_radio(
    artist: &str,
    title: &str,
    n: usize,
    weight_audio: f32,
    json: bool,
) -> Result<i32> {
    let results = recommend::radio_playlist(artist, title, n, weight_audio)?;
    if results.is_empty() {
        return Ok(2);
    }
    output_results(&results, json);
    Ok(0)
}

fn cmd_profile(json: bool) -> Result<i32> {
    let profile = recommend::taste_profile()?;

    if json {
        println!("{}", serde_json::to_string(&profile)?);
    } else {
        eprintln!("Library: {} tracks\n", profile.total_tracks);

        eprintln!("Top Genres:");
        for (genre, count) in &profile.top_genres {
            eprintln!("  {genre}: {count}");
        }

        eprintln!("\nTop Artists:");
        for (artist, count) in &profile.top_artists {
            eprintln!("  {artist}: {count}");
        }

        eprintln!("\nDecades:");
        for (decade, count) in &profile.decades {
            eprintln!("  {decade}: {count}");
        }
    }

    Ok(0)
}

fn cmd_serve(port: u16, bootstrap: Option<String>) -> Result<i32> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

        let node_handle = tokio::spawn(async move {
            if let Err(e) = net::node::run_node(port, bootstrap, event_tx).await {
                eprintln!("Node error: {e:#}");
            }
        });

        // Handle node events
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                match event {
                    net::node::NodeEvent::PeerConnected(peer) => {
                        eprintln!("Peer connected: {peer}");
                    }
                    net::node::NodeEvent::PeerDisconnected(peer) => {
                        eprintln!("Peer disconnected: {peer}");
                    }
                    net::node::NodeEvent::SearchRequest {
                        peer,
                        channel,
                        query,
                    } => {
                        eprintln!("Search request from {peer}");
                        let response = net::node::handle_search_query(&query);
                        eprintln!(
                            "Would respond with {} results",
                            response.results.len()
                        );
                        drop(channel);
                    }
                    net::node::NodeEvent::SearchResponse { peer, response } => {
                        eprintln!(
                            "Search response from {peer}: {} results",
                            response.results.len()
                        );
                    }
                }
            }
        });

        node_handle.await?;
        Ok(0)
    })
}

fn cmd_peers() -> i32 {
    eprintln!("Peers command requires a running 'mzk serve' instance.");
    eprintln!("(IPC between CLI and daemon not yet implemented)");
    1
}

fn cmd_network_search(_query: &str, _n: usize, _json: bool) -> i32 {
    eprintln!("Network search requires a running 'mzk serve' instance.");
    eprintln!("(IPC between CLI and daemon not yet implemented)");
    1
}

fn cmd_network_similar(
    _artist: &str,
    _title: &str,
    _n: usize,
    _weight_audio: f32,
    _json: bool,
) -> i32 {
    eprintln!("Network similar requires a running 'mzk serve' instance.");
    eprintln!("(IPC between CLI and daemon not yet implemented)");
    1
}

fn output_results(results: &[SearchResult], json: bool) {
    for r in results {
        if json {
            if let Ok(j) = serde_json::to_string(r) {
                println!("{j}");
            }
        } else {
            println!("{}", r.to_tsv());
        }
    }
}
