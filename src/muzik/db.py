"""LanceDB schema and read/write operations."""

from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

DB_PATH = Path.home() / ".local" / "share" / "muzik"

METADATA_DIM = 384
AUDIO_DIM = 512

TRACKS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("artist", pa.string()),
    pa.field("album_artist", pa.string()),
    pa.field("album", pa.string()),
    pa.field("genre", pa.string()),
    pa.field("year", pa.int32()),
    pa.field("track_num", pa.int32()),
    pa.field("filepath", pa.string()),
    pa.field("metadata_vec", pa.list_(pa.float32(), METADATA_DIM)),
    pa.field("audio_vec", pa.list_(pa.float32(), AUDIO_DIM)),
])

ALBUMS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("artist", pa.string()),
    pa.field("album", pa.string()),
    pa.field("year", pa.int32()),
    pa.field("genres", pa.string()),
    pa.field("track_count", pa.int32()),
    pa.field("metadata_vec", pa.list_(pa.float32(), METADATA_DIM)),
    pa.field("audio_vec", pa.list_(pa.float32(), AUDIO_DIM)),
])


def get_db() -> lancedb.DBConnection:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def _zero_vec(dim: int) -> list[float]:
    return [0.0] * dim


def get_or_create_tracks(db: lancedb.DBConnection) -> lancedb.table.Table:
    if "tracks" in db.table_names():
        return db.open_table("tracks")
    return db.create_table("tracks", schema=TRACKS_SCHEMA)


def get_or_create_albums(db: lancedb.DBConnection) -> lancedb.table.Table:
    if "albums" in db.table_names():
        return db.open_table("albums")
    return db.create_table("albums", schema=ALBUMS_SCHEMA)


def upsert_tracks(table: lancedb.table.Table, rows: list[dict]) -> None:
    """Insert or overwrite tracks by id via merge-insert."""
    if not rows:
        return
    for row in rows:
        row.setdefault("metadata_vec", _zero_vec(METADATA_DIM))
        row.setdefault("audio_vec", _zero_vec(AUDIO_DIM))
    table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)


def upsert_albums(table: lancedb.table.Table, rows: list[dict]) -> None:
    """Insert or overwrite albums by id via merge-insert."""
    if not rows:
        return
    for row in rows:
        row.setdefault("metadata_vec", _zero_vec(METADATA_DIM))
        row.setdefault("audio_vec", _zero_vec(AUDIO_DIM))
    table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)


def all_tracks(table: lancedb.table.Table) -> pa.Table:
    return table.to_arrow()


def all_albums(table: lancedb.table.Table) -> pa.Table:
    return table.to_arrow()


def track_count(table: lancedb.table.Table) -> int:
    return table.count_rows()


def album_count(table: lancedb.table.Table) -> int:
    return table.count_rows()


def _escape_sql(s: str) -> str:
    return s.replace("'", "''")


def find_track(table: lancedb.table.Table, artist: str, title: str) -> dict | None:
    """Find a track by artist and title (case-insensitive)."""
    arrow = table.to_lance().scanner(
        filter=f"lower(artist) = '{_escape_sql(artist.lower())}' "
               f"AND lower(title) = '{_escape_sql(title.lower())}'"
    ).to_table()
    if len(arrow) == 0:
        return None
    return arrow.slice(0, 1).to_pylist()[0]


def find_album(table: lancedb.table.Table, artist: str, album: str) -> dict | None:
    """Find an album by artist and album name (case-insensitive)."""
    arrow = table.to_lance().scanner(
        filter=f"lower(artist) = '{_escape_sql(artist.lower())}' "
               f"AND lower(album) = '{_escape_sql(album.lower())}'"
    ).to_table()
    if len(arrow) == 0:
        return None
    return arrow.slice(0, 1).to_pylist()[0]
