"""Matching: offset voting, fully vectorised with numpy."""
from dataclasses import dataclass, field
import numpy as np
from .database import FingerprintDB
from .fingerprint import fingerprint, HOP, SR

MIN_SCORE = 10       # best song needs at least this many aligned hashes (tune with evaluate.py)
MIN_RATIO = 1.5      # ...and best_score / second_best_score >= this
FRAME_SEC = HOP / SR


@dataclass
class MatchResult:
    matched: bool
    message: str
    confidence: float = 0.0                 # best score / second-best score
    candidates: list = field(default_factory=list)   # top-3 dicts
    hist_offsets_sec: np.ndarray = None     # for the offset-histogram plot
    hist_counts: np.ndarray = None

    @property
    def best(self):
        return self.candidates[0] if self.candidates else None


def _lookup(db: FingerprintDB, q_hashes: np.ndarray):
    """For every query hash, find ALL database rows with the same hash.

    Returns (row_indices, query_index_for_each_row).
    """
    lo = np.searchsorted(db.hashes, q_hashes, side="left")
    hi = np.searchsorted(db.hashes, q_hashes, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    q_idx = np.repeat(np.arange(len(q_hashes)), counts)
    start_of_group = np.repeat(np.cumsum(counts) - counts, counts)
    rows = np.arange(total) - start_of_group + np.repeat(lo, counts)
    return rows, q_idx


def identify(y: np.ndarray, db: FingerprintDB, top_k: int = 3) -> MatchResult:
    q_hashes, q_times = fingerprint(y)
    if len(q_hashes) == 0:
        return MatchResult(False, "Not enough distinct sound features in this clip.")

    rows, q_idx = _lookup(db, q_hashes)
    if len(rows) == 0:
        return MatchResult(False, "No matching fingerprints found in the database.")

    # Time offset between song and query. For the TRUE song, many hashes agree
    # on the same offset -> a spike. For wrong songs offsets are random.
    t_db = db.times[rows].astype(np.int64)
    t_q = q_times[q_idx].astype(np.int64)
    shift = int(t_q.max())
    offset = t_db - t_q + shift                    # >= 0
    span = int(offset.max()) + 1
    song = db.song_ids[rows].astype(np.int64)

    keys, votes = np.unique(song * span + offset, return_counts=True)
    key_song, key_off = keys // span, keys % span

    # best offset per song: sort by votes (desc), keep first row of each song
    order = np.argsort(-votes, kind="stable")
    uniq_songs, first = np.unique(key_song[order], return_index=True)
    best_rows = order[first]
    song_scores = votes[best_rows]
    song_offsets = key_off[best_rows]
    rank = np.argsort(-song_scores, kind="stable")[:top_k]

    candidates = []
    for r in rank:
        sid = int(uniq_songs[r])
        candidates.append({
            "song_id": sid,
            "name": str(db.names[sid]),
            "genre": str(db.genres[sid]),
            "path": str(db.paths[sid]),
            "score": int(song_scores[r]),
            "position_sec": round((int(song_offsets[r]) - shift) * FRAME_SEC, 2),
        })

    best = candidates[0]["score"]
    second = candidates[1]["score"] if len(candidates) > 1 else 0
    confidence = best / max(second, 1)
    matched = best >= MIN_SCORE and confidence >= MIN_RATIO

    # histogram of offsets for the best song (the "spike" plot)
    sel = song == candidates[0]["song_id"]
    counts = np.bincount(offset[sel])
    xs = (np.arange(len(counts)) - shift) * FRAME_SEC

    msg = "Match found." if matched else "No confident match found."
    return MatchResult(matched, msg, round(float(confidence), 2), candidates, xs, counts)
