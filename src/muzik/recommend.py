"""Hybrid similarity queries and recommendation engine."""

from collections import Counter

import numpy as np

from muzik.db import (
    all_albums,
    all_tracks,
    find_album,
    find_track,
    get_db,
    get_or_create_albums,
    get_or_create_tracks,
)

DEFAULT_WEIGHT_AUDIO = 0.7
CLAP_CKPT = "~/.local/share/muzik/models/630k-audioset-best.pt"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _is_zero(vec: list[float] | None) -> bool:
    if vec is None:
        return True
    return all(v == 0.0 for v in vec[:8])


def _hybrid_score(
    query_meta: np.ndarray | None,
    query_audio: np.ndarray | None,
    cand_meta: list[float] | None,
    cand_audio: list[float] | None,
    weight_audio: float,
) -> float:
    """Compute hybrid similarity score from metadata and audio vectors."""
    meta_score = 0.0
    audio_score = 0.0
    has_meta = query_meta is not None and not _is_zero(cand_meta)
    has_audio = query_audio is not None and not _is_zero(cand_audio)

    if has_meta:
        meta_score = _cosine_sim(query_meta, np.array(cand_meta))
    if has_audio:
        audio_score = _cosine_sim(query_audio, np.array(cand_audio))

    # Adjust weights based on what's available
    if has_audio and has_meta:
        return (1 - weight_audio) * meta_score + weight_audio * audio_score
    elif has_audio:
        return audio_score
    elif has_meta:
        return meta_score
    return 0.0


def similar_tracks(
    artist: str,
    title: str,
    n: int = 10,
    weight_audio: float = DEFAULT_WEIGHT_AUDIO,
) -> list[dict]:
    """Find tracks most similar to the given track."""
    db = get_db()
    tracks_table = get_or_create_tracks(db)

    query = find_track(tracks_table, artist, title)
    if query is None:
        raise ValueError(f"Track not found: {artist} - {title}")

    query_meta = np.array(query["metadata_vec"]) if not _is_zero(query.get("metadata_vec")) else None
    query_audio = np.array(query["audio_vec"]) if not _is_zero(query.get("audio_vec")) else None

    if query_meta is None and query_audio is None:
        raise RuntimeError("Track has no embeddings. Run 'muzik embed' first.")

    rows = all_tracks(tracks_table).to_pylist()

    scored = []
    for row in rows:
        if row["id"] == query["id"]:
            continue
        score = _hybrid_score(
            query_meta, query_audio,
            row.get("metadata_vec"), row.get("audio_vec"),
            weight_audio,
        )
        scored.append({**row, "_score": score})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:n]


def similar_albums(
    artist: str,
    album: str,
    n: int = 10,
    weight_audio: float = DEFAULT_WEIGHT_AUDIO,
) -> list[dict]:
    """Find albums most similar to the given album."""
    db = get_db()
    albums_table = get_or_create_albums(db)

    query = find_album(albums_table, artist, album)
    if query is None:
        raise ValueError(f"Album not found: {artist} - {album}")

    query_meta = np.array(query["metadata_vec"]) if not _is_zero(query.get("metadata_vec")) else None
    query_audio = np.array(query["audio_vec"]) if not _is_zero(query.get("audio_vec")) else None

    if query_meta is None and query_audio is None:
        raise RuntimeError("Album has no embeddings. Run 'muzik embed' first.")

    rows = all_albums(albums_table).to_pylist()

    scored = []
    for row in rows:
        if row["id"] == query["id"]:
            continue
        score = _hybrid_score(
            query_meta, query_audio,
            row.get("metadata_vec"), row.get("audio_vec"),
            weight_audio,
        )
        scored.append({**row, "_score": score})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:n]


def taste_profile() -> dict:
    """Generate a taste profile summary from the library."""
    db = get_db()
    tracks_table = get_or_create_tracks(db)
    rows = all_tracks(tracks_table).to_pylist()

    genre_counter: Counter[str] = Counter()
    artist_counter: Counter[str] = Counter()
    decade_counter: Counter[str] = Counter()

    for row in rows:
        if row["genre"]:
            genre_counter[row["genre"]] += 1
        if row["artist"]:
            artist_counter[row["artist"]] += 1
        if row["year"] and row["year"] > 0:
            decade = f"{(row['year'] // 10) * 10}s"
            decade_counter[decade] += 1

    return {
        "total_tracks": len(rows),
        "top_genres": genre_counter.most_common(15),
        "top_artists": artist_counter.most_common(15),
        "decades": sorted(decade_counter.items(), key=lambda x: x[0]),
    }


def search_by_text(
    query: str,
    n: int = 10,
) -> list[dict]:
    """Search library using a natural language description via CLAP text embeddings.

    Embeds the query text into CLAP's shared audio-text space and finds
    tracks whose audio embeddings are closest to the description.
    """
    import contextlib
    import logging
    import os
    import sys
    import warnings
    from pathlib import Path

    # Suppress all CLAP/torch loading noise
    logging.disable(logging.CRITICAL)
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch

            import laion_clap

            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
            ckpt = Path(CLAP_CKPT).expanduser()
            if ckpt.exists():
                model.load_ckpt(ckpt=str(ckpt))
            else:
                model.load_ckpt()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()
        logging.disable(logging.NOTSET)
    model.eval()

    with torch.no_grad():
        text_embed = model.get_text_embedding([query], use_tensor=False)
    query_vec = text_embed[0]
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    db = get_db()
    tracks_table = get_or_create_tracks(db)
    rows = all_tracks(tracks_table).to_pylist()

    scored = []
    for row in rows:
        if _is_zero(row.get("audio_vec")):
            continue
        score = _cosine_sim(query_vec, np.array(row["audio_vec"]))
        scored.append({**row, "_score": score})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:n]


def radio_playlist(
    artist: str,
    title: str,
    n: int = 20,
    weight_audio: float = DEFAULT_WEIGHT_AUDIO,
) -> list[dict]:
    """Generate a radio-style playlist seeded from a track.

    Uses iterative expansion: finds similar tracks, then picks some
    similar to those, creating a more diverse playlist.
    """
    # Start with the most similar tracks
    initial = similar_tracks(artist, title, n=n * 2, weight_audio=weight_audio)

    if len(initial) <= n:
        return initial

    # Pick top half directly, then diversify
    playlist = initial[: n // 2]
    seen_ids = {t["id"] for t in playlist}

    # Add remaining tracks, preferring variety in artists
    seen_artists: set[str] = {t["artist"] for t in playlist}
    remaining = [t for t in initial[n // 2 :] if t["id"] not in seen_ids]

    # Prioritize unseen artists for diversity
    remaining.sort(
        key=lambda t: (t["artist"] in seen_artists, -t["_score"])
    )

    for track in remaining:
        if len(playlist) >= n:
            break
        playlist.append(track)
        seen_ids.add(track["id"])
        seen_artists.add(track["artist"])

    return playlist
