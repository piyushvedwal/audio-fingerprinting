"""Cut a few short demo clips for the 'Try a sample clip' buttons.

IMPORTANT: samples/ is committed to a PUBLIC repo. Only use tracks you are
allowed to share (e.g. CC-licensed tracks from FMA, or your own music).

    python scripts/make_samples.py --db db/fingerprints.npz --data data/ --n 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.database import load_db                        # noqa: E402
from src.fingerprint import SR                          # noqa: E402
from src.utils import add_noise, load_audio_file        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="db/fingerprints.npz")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--snr", type=float, default=15, help="mild noise so the demo feels real")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="samples")
    args = ap.parse_args()

    db = load_db(args.db)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    for k, sid in enumerate(rng.choice(db.n_songs, size=min(args.n, db.n_songs), replace=False), 1):
        offset = float(rng.uniform(2, 20))
        y = load_audio_file(Path(args.data) / str(db.paths[sid]), duration=args.seconds, offset=offset)
        y = add_noise(y, args.snr, rng)
        y = y / max(np.max(np.abs(y)), 1e-9) * 0.9
        wavfile.write(out / f"sample_{k}.wav", SR, (y * 32767).astype(np.int16))
        print(f"sample_{k}.wav  <-  {db.names[sid]}")


if __name__ == "__main__":
    main()
