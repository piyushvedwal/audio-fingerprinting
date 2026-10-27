"""Run with:  pytest -q   (uses synthetic 'songs', no dataset needed)"""
import numpy as np
import pytest

from src.database import build_from_fingerprints, load_db, save_db
from src.fingerprint import SR, fingerprint
from src.match import identify
from src.utils import add_noise

N_SONGS = 15


def synth_song(seed, seconds=30):
    """Random melody: overlapping notes with harmonics + faint noise."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR), np.float32)
    pos = 0
    while pos < len(out):
        n = int(rng.uniform(0.15, 0.5) * SR)
        f0 = rng.uniform(150, 1500)
        t = np.arange(n) / SR
        note = sum((1 / (k + 1)) * np.sin(2 * np.pi * f0 * (k + 1) * t) for k in range(4)) * np.hanning(n)
        out[pos:pos + n] += note[: len(out) - pos].astype(np.float32)
        pos += n // 2
    return out + 0.01 * rng.standard_normal(len(out)).astype(np.float32)


@pytest.fixture(scope="module")
def world():
    songs = [synth_song(s) for s in range(N_SONGS)]
    fps = [fingerprint(y) for y in songs]
    db = build_from_fingerprints(fps, [f"s{i}" for i in range(N_SONGS)], ["g"] * N_SONGS,
                                 [f"g/s{i}.wav" for i in range(N_SONGS)])
    return songs, db


def test_fingerprint_is_deterministic():
    y = synth_song(0, seconds=10)
    h1, t1 = fingerprint(y)
    h2, t2 = fingerprint(y)
    assert len(h1) > 0
    assert np.array_equal(h1, h2) and np.array_equal(t1, t2)


def test_hashes_fit_in_30_bits():
    h, _ = fingerprint(synth_song(1, seconds=10))
    assert h.max() < 2 ** 30


def test_self_match(world):
    songs, db = world
    for i in (0, 4, 9):
        clip = songs[i][7 * SR: 12 * SR]            # 5 s from the middle
        res = identify(clip, db)
        assert res.matched and res.best["name"] == f"s{i}"
        assert abs(res.best["position_sec"] - 7.0) < 0.3


def test_noisy_match(world):
    songs, db = world
    rng = np.random.default_rng(0)
    ok = 0
    for i in range(N_SONGS):
        clip = add_noise(songs[i][3 * SR: 11 * SR], 10, rng)   # 8 s at 10 dB SNR
        res = identify(clip, db)
        ok += res.matched and res.best["name"] == f"s{i}"
    assert ok >= 0.8 * N_SONGS


def test_save_and_load_roundtrip(world, tmp_path):
    _, db = world
    p = tmp_path / "db.npz"
    save_db(db, str(p))
    db2 = load_db(str(p))
    assert np.array_equal(db.hashes, db2.hashes) and list(db.names) == list(db2.names)
