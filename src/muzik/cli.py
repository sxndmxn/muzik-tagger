"""CLI entrypoint for muzik-tagger."""

from pathlib import Path

import typer

app = typer.Typer(
    name="muzik",
    help="Vector music recommendation system",
    no_args_is_help=True,
)


@app.command()
def scan(
    library: Path = typer.Argument(..., help="Path to music library directory"),
) -> None:
    """Scan a music library and extract metadata from audio tags."""
    if not library.is_dir():
        typer.echo(f"Error: {library} is not a directory", err=True)
        raise typer.Exit(1)

    from muzik.scan import scan_library

    typer.echo(f"Scanning {library}...")
    tracks, albums = scan_library(library)
    typer.echo(f"Done: {tracks} tracks, {albums} albums")


@app.command()
def embed(
    metadata: bool = typer.Option(False, "--metadata", help="Generate metadata text embeddings"),
    audio: bool = typer.Option(False, "--audio", help="Generate CLAP audio embeddings (GPU)"),
    all_: bool = typer.Option(False, "--all", help="Generate both metadata and audio embeddings"),
) -> None:
    """Generate vector embeddings for tracks."""
    if not (metadata or audio or all_):
        typer.echo("Specify --metadata, --audio, or --all", err=True)
        raise typer.Exit(1)

    if metadata or all_:
        from muzik.embed_metadata import embed_metadata

        typer.echo("Generating metadata embeddings...")
        count = embed_metadata()
        typer.echo(f"Embedded metadata for {count} tracks")

    if audio or all_:
        from muzik.embed_audio import embed_audio

        typer.echo("Generating CLAP audio embeddings...")
        count = embed_audio()
        if count == 0:
            typer.echo("All tracks already have audio embeddings")
        else:
            typer.echo(f"Embedded audio for {count} tracks")


@app.command()
def similar(
    artist: str = typer.Argument(..., help="Artist name"),
    name: str = typer.Argument(..., help="Track title or album name"),
    album: bool = typer.Option(False, "--album", help="Search for similar albums instead of tracks"),
    track: bool = typer.Option(False, "--track", help="Search for similar tracks (default)"),
    n: int = typer.Option(10, "-n", help="Number of results"),
    weight_audio: float = typer.Option(0.7, "--weight-audio", "-w", help="Audio weight (0-1)"),
) -> None:
    """Find similar tracks or albums."""
    from muzik.recommend import similar_albums, similar_tracks

    if album:
        results = similar_albums(artist, name, n=n, weight_audio=weight_audio)
        typer.echo(f"\nAlbums similar to: {artist} - {name}\n")
        for i, r in enumerate(results, 1):
            score = r["_score"]
            typer.echo(f"  {i:2d}. {r['artist']} - {r['album']} ({r['year']}) [{score:.3f}]")
    else:
        results = similar_tracks(artist, name, n=n, weight_audio=weight_audio)
        typer.echo(f"\nTracks similar to: {artist} - {name}\n")
        for i, r in enumerate(results, 1):
            score = r["_score"]
            typer.echo(f"  {i:2d}. {r['artist']} - {r['title']} ({r['album']}) [{score:.3f}]")


@app.command()
def profile() -> None:
    """Show a taste profile summary of your library."""
    from muzik.recommend import taste_profile

    p = taste_profile()
    typer.echo(f"\nLibrary: {p['total_tracks']} tracks\n")

    typer.echo("Top Genres:")
    for genre, count in p["top_genres"]:
        typer.echo(f"  {genre}: {count}")

    typer.echo("\nTop Artists:")
    for artist, count in p["top_artists"]:
        typer.echo(f"  {artist}: {count}")

    typer.echo("\nDecades:")
    for decade, count in p["decades"]:
        typer.echo(f"  {decade}: {count}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language description of what you want to hear"),
    n: int = typer.Option(10, "-n", help="Number of results"),
) -> None:
    """Search your library by description (e.g. 'dark ambient drone', 'aggressive breakbeats')."""
    from muzik.recommend import search_by_text

    typer.echo(f"Searching for: {query}\n")
    results = search_by_text(query, n=n)
    if not results:
        typer.echo("No results (audio embeddings may not be generated yet)")
        raise typer.Exit(1)
    for i, r in enumerate(results, 1):
        score = r["_score"]
        typer.echo(f"  {i:2d}. {r['artist']} - {r['title']} ({r['album']}) [{score:.3f}]")


@app.command()
def radio(
    artist: str = typer.Argument(..., help="Seed artist"),
    title: str = typer.Argument(..., help="Seed track title"),
    n: int = typer.Option(20, "-n", help="Playlist length"),
    weight_audio: float = typer.Option(0.7, "--weight-audio", "-w", help="Audio weight (0-1)"),
) -> None:
    """Generate a radio playlist seeded from a track."""
    from muzik.recommend import radio_playlist

    playlist = radio_playlist(artist, title, n=n, weight_audio=weight_audio)
    typer.echo(f"\nRadio from: {artist} - {title}\n")
    for i, t in enumerate(playlist, 1):
        typer.echo(f"  {i:2d}. {t['artist']} - {t['title']} ({t['album']})")


if __name__ == "__main__":
    app()
