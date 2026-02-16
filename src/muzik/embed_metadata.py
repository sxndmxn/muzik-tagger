"""Generate text embeddings for track metadata using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from muzik.db import (
    all_tracks,
    get_db,
    get_or_create_albums,
    get_or_create_tracks,
    upsert_albums,
    upsert_tracks,
)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def _build_metadata_text(row: dict) -> str:
    """Build the text string used for embedding."""
    parts = [row["artist"], row["title"], row["album"], row["genre"]]
    if row["year"] and row["year"] > 0:
        parts.append(str(row["year"]))
    return " ".join(p for p in parts if p)


def embed_metadata() -> int:
    """Generate metadata embeddings for all tracks. Returns count of embedded tracks."""
    db = get_db()
    tracks_table = get_or_create_tracks(db)
    arrow = all_tracks(tracks_table)

    if len(arrow) == 0:
        raise RuntimeError("No tracks in database. Run 'muzik scan' first.")

    rows = arrow.to_pylist()
    texts = [_build_metadata_text(r) for r in rows]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)

    # Update tracks with metadata vectors
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        batch_vecs = embeddings[i : i + BATCH_SIZE]
        for row, vec in zip(batch, batch_vecs):
            row["metadata_vec"] = vec.tolist()
        upsert_tracks(tracks_table, batch)

    # Aggregate into albums
    _aggregate_albums(db, rows, embeddings)

    return len(rows)


def _aggregate_albums(
    db: object, rows: list[dict], embeddings: np.ndarray
) -> None:
    """Compute album-level metadata vectors as mean of track vectors."""
    albums_table = get_or_create_albums(db)
    album_arrow = albums_table.to_arrow()
    album_list = album_arrow.to_pylist()

    # Group track embeddings by album
    album_vecs: dict[str, list[np.ndarray]] = {}
    for row, vec in zip(rows, embeddings):
        key = f"{row['album_artist'] or row['artist']}||{row['album']}".lower().strip()
        album_vecs.setdefault(key, []).append(vec)

    # Match to existing albums and update
    updates = []
    for album in album_list:
        key = f"{album['artist']}||{album['album']}".lower().strip()
        vecs = album_vecs.get(key)
        if vecs:
            album["metadata_vec"] = np.mean(vecs, axis=0).tolist()
            updates.append(album)

    if updates:
        for i in range(0, len(updates), BATCH_SIZE):
            upsert_albums(albums_table, updates[i : i + BATCH_SIZE])
