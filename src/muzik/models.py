"""Data models for tracks and albums."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    id: str
    title: str
    artist: str
    album_artist: str
    album: str
    genre: str
    year: int
    track_num: int
    filepath: str
    metadata_vec: np.ndarray | None = field(default=None, repr=False)
    audio_vec: np.ndarray | None = field(default=None, repr=False)

    @property
    def metadata_text(self) -> str:
        """Text string used for metadata embedding."""
        parts = [self.artist, self.title, self.album, self.genre, str(self.year)]
        return " ".join(p for p in parts if p and p != "0")


@dataclass
class Album:
    id: str
    artist: str
    album: str
    year: int
    genres: str
    track_count: int
    metadata_vec: np.ndarray | None = field(default=None, repr=False)
    audio_vec: np.ndarray | None = field(default=None, repr=False)
