"""
Compact fingerprint database.

Memory / size trade-off
-----------------------
A Python dict {hash: [(song, time), ...]} costs ~100+ bytes per entry
(object headers, tuples, lists) -> 6M entries would need > 600 MB: too much
for a free 1 GB server. Instead we keep THREE flat numpy arrays sorted by hash:

    hashes   uint32  (4 bytes)
    song_ids uint16  (2 bytes)
    times    uint16  (2 bytes)        -> 8 bytes per entry (~12x smaller)

Lookup uses np.searchsorted (binary search) on the sorted hash array, and the
file is stored with np.savez_compressed (typically another 30-50% smaller).
Limits: <= 65,535 songs and <= 65,535 frames (~25 min) per song.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class FingerprintDB:
    hashes: np.ndarray      # uint32, sorted ascending
    song_ids: np.ndarray    # uint16, aligned with hashes
    times: np.ndarray       # uint16, aligned with hashes
    names: np.ndarray       # str, one per song  (e.g. "blues.00000")
    genres: np.ndarray      # str, one per song  (e.g. "blues")
    paths: np.ndarray       # str, one per song  (relative path, used by evaluate.py)

    @property
    def n_songs(self) -> int:
        return len(self.names)

    @property
    def n_hashes(self) -> int:
        return len(self.hashes)


def build_from_fingerprints(fps, names, genres, paths) -> FingerprintDB:
    """fps: list of (hashes, times) tuples, one per song (same order as names)."""
    if not fps:
        raise ValueError("No fingerprints to build a database from.")
    hashes = np.concatenate([h for h, _ in fps]).astype(np.uint32)
    times = np.concatenate([t for _, t in fps])
    song_ids = np.concatenate(
        [np.full(len(h), i, dtype=np.uint16) for i, (h, _) in enumerate(fps)])
    if times.max() > 65535:
        raise ValueError("A song is longer than 65,535 frames; shorten it.")
    order = np.argsort(hashes, kind="stable")            # sort once, offline
    return FingerprintDB(hashes[order], song_ids[order],
                         times[order].astype(np.uint16),
                         np.array(names), np.array(genres), np.array(paths))


def save_db(db: FingerprintDB, path: str) -> None:
    np.savez_compressed(path, hashes=db.hashes, song_ids=db.song_ids,
                        times=db.times, names=db.names, genres=db.genres,
                        paths=db.paths)


def load_db(path: str) -> FingerprintDB:
    with np.load(path, allow_pickle=False) as d:
        return FingerprintDB(d["hashes"], d["song_ids"], d["times"],
                             d["names"], d["genres"], d["paths"])
