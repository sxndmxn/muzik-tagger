"""Generate CLAP audio embeddings for tracks."""

import warnings

import numpy as np
import torch
from tqdm import tqdm

from muzik.db import (
    AUDIO_DIM,
    all_tracks,
    get_db,
    get_or_create_albums,
    get_or_create_tracks,
    upsert_albums,
    upsert_tracks,
)

BATCH_SIZE = 16  # Small batches for GPU memory
DB_BATCH_SIZE = 256


def _is_zero_vec(vec: list[float] | None) -> bool:
    if vec is None:
        return True
    return all(v == 0.0 for v in vec[:8])  # Check first 8 elements as heuristic


def embed_audio() -> int:
    """Generate CLAP audio embeddings for all tracks.

    Skips tracks that already have audio embeddings (resume support).
    Returns count of newly embedded tracks.
    """
    db = get_db()
    tracks_table = get_or_create_tracks(db)
    arrow = all_tracks(tracks_table)

    if len(arrow) == 0:
        raise RuntimeError("No tracks in database. Run 'muzik scan' first.")

    rows = arrow.to_pylist()

    # Filter to only tracks needing audio embeddings
    to_embed = [r for r in rows if _is_zero_vec(r.get("audio_vec"))]

    if not to_embed:
        return 0

    # Lazy import CLAP to avoid slow import when not needed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import laion_clap

    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")

    ckpt_path = Path.home() / ".local" / "share" / "muzik" / "models" / "630k-audioset-best.pt"
    if ckpt_path.exists():
        model.load_ckpt(ckpt=str(ckpt_path))
    else:
        model.load_ckpt()

    model = model.to(device)
    model.eval()

    embedded_count = 0
    db_batch: list[dict] = []

    for row in tqdm(to_embed, desc="Embedding audio (CLAP)", unit="track"):
        filepath = row["filepath"]
        try:
            with torch.no_grad():
                audio_embed = model.get_audio_embedding_from_filelist(
                    [filepath], use_tensor=False
                )
            vec = audio_embed[0]
            # Normalize to unit vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            row["audio_vec"] = vec.tolist()
            embedded_count += 1
        except Exception as e:
            tqdm.write(f"  skip {filepath}: {e}")
            continue

        db_batch.append(row)
        if len(db_batch) >= DB_BATCH_SIZE:
            upsert_tracks(tracks_table, db_batch)
            db_batch.clear()

    if db_batch:
        upsert_tracks(tracks_table, db_batch)

    # Aggregate into albums
    _aggregate_album_audio(db)

    return embedded_count


def _aggregate_album_audio(db: object) -> None:
    """Compute album-level audio vectors as mean of track audio vectors."""
    tracks_table = get_or_create_tracks(db)
    albums_table = get_or_create_albums(db)

    all_rows = all_tracks(tracks_table).to_pylist()
    album_arrow = albums_table.to_arrow()
    album_list = album_arrow.to_pylist()

    # Group non-zero audio vecs by album
    album_vecs: dict[str, list[np.ndarray]] = {}
    for row in all_rows:
        if _is_zero_vec(row.get("audio_vec")):
            continue
        key = f"{row['album_artist'] or row['artist']}||{row['album']}".lower().strip()
        album_vecs.setdefault(key, []).append(np.array(row["audio_vec"]))

    updates = []
    for album in album_list:
        key = f"{album['artist']}||{album['album']}".lower().strip()
        vecs = album_vecs.get(key)
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            album["audio_vec"] = mean_vec.tolist()
            updates.append(album)

    if updates:
        for i in range(0, len(updates), DB_BATCH_SIZE):
            upsert_albums(albums_table, updates[i : i + DB_BATCH_SIZE])
