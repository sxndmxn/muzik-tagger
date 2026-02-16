"""Scan music library and extract metadata from audio tags."""

import hashlib
from collections import defaultdict
from pathlib import Path

from mutagen import File as MutagenFile
from tqdm import tqdm

from muzik.db import (
    get_db,
    get_or_create_albums,
    get_or_create_tracks,
    upsert_albums,
    upsert_tracks,
)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".ogg", ".opus", ".m4a", ".wma", ".wav", ".aiff"}
BATCH_SIZE = 500


def _track_id(filepath: str) -> str:
    return hashlib.sha256(filepath.encode()).hexdigest()[:16]


def _album_id(artist: str, album: str) -> str:
    key = f"{artist.lower().strip()}||{album.lower().strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get_tag(audio: MutagenFile, keys: list[str], default: str = "") -> str:
    """Extract a tag value, trying multiple key names for format compatibility."""
    for key in keys:
        val = audio.get(key)
        if val:
            # mutagen returns lists for most tag types
            if isinstance(val, list):
                return str(val[0])
            return str(val)
    return default


def _get_int_tag(audio: MutagenFile, keys: list[str], default: int = 0) -> int:
    raw = _get_tag(audio, keys, "")
    if not raw:
        return default
    # Handle "3/12" style track numbers
    raw = raw.split("/")[0].strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _extract_metadata(filepath: Path) -> dict | None:
    """Extract metadata from a single audio file."""
    try:
        audio = MutagenFile(str(filepath), easy=True)
    except Exception:
        return None

    if audio is None:
        return None

    artist = _get_tag(audio, ["artist", "albumartist", "performer"])
    album_artist = _get_tag(audio, ["albumartist", "artist", "performer"])
    album = _get_tag(audio, ["album"])
    title = _get_tag(audio, ["title"])
    genre = _get_tag(audio, ["genre"])
    year = _get_int_tag(audio, ["date", "year", "originaldate"])
    track_num = _get_int_tag(audio, ["tracknumber"])

    # Fallback: infer artist/album from directory structure (Artist/Album/track.ext)
    if not artist and filepath.parent.parent.name:
        artist = filepath.parent.parent.name
    if not album and filepath.parent.name:
        album = filepath.parent.name
    if not title:
        title = filepath.stem

    return {
        "id": _track_id(str(filepath)),
        "title": title,
        "artist": artist,
        "album_artist": album_artist or artist,
        "album": album,
        "genre": genre,
        "year": year,
        "track_num": track_num,
        "filepath": str(filepath),
    }


def scan_library(library_path: Path) -> tuple[int, int]:
    """Scan a music library directory and store metadata in LanceDB.

    Returns (track_count, album_count).
    """
    # Collect all audio files
    audio_files = [
        p for p in library_path.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
    ]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {library_path}")

    db = get_db()
    tracks_table = get_or_create_tracks(db)

    # Extract metadata and batch-upsert
    batch: list[dict] = []
    total_tracks = 0
    album_data: dict[str, dict] = {}

    for filepath in tqdm(audio_files, desc="Scanning tracks", unit="file"):
        meta = _extract_metadata(filepath)
        if meta is None:
            continue

        batch.append(meta)
        total_tracks += 1

        # Collect album info
        aid = _album_id(meta["artist"], meta["album"])
        if aid not in album_data:
            album_data[aid] = {
                "id": aid,
                "artist": meta["album_artist"] or meta["artist"],
                "album": meta["album"],
                "year": meta["year"],
                "genres": defaultdict(int),
                "track_count": 0,
            }
        album_data[aid]["track_count"] += 1
        if meta["genre"]:
            album_data[aid]["genres"][meta["genre"]] += 1

        if len(batch) >= BATCH_SIZE:
            upsert_tracks(tracks_table, batch)
            batch.clear()

    if batch:
        upsert_tracks(tracks_table, batch)

    # Build and upsert albums
    albums_table = get_or_create_albums(db)
    album_rows = []
    for info in album_data.values():
        genres = info["genres"]
        # Sort genres by frequency, join
        sorted_genres = sorted(genres, key=genres.get, reverse=True)  # type: ignore[arg-type]
        album_rows.append({
            "id": info["id"],
            "artist": info["artist"],
            "album": info["album"],
            "year": info["year"],
            "genres": ", ".join(sorted_genres),
            "track_count": info["track_count"],
        })

    # Batch upsert albums
    for i in range(0, len(album_rows), BATCH_SIZE):
        upsert_albums(albums_table, album_rows[i : i + BATCH_SIZE])

    return total_tracks, len(album_data)
