"""Measure accuracy for different clip lengths and noise levels.

Needs the RAW audio (local only):
    python scripts/evaluate.py --db db/fingerprints.npz --data data/genres_original
Outputs: assets/results.csv, assets/heatmap.png, assets/accuracy_curves.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.database import load_db                         # noqa: E402
from src.fingerprint import SR                           # noqa: E402
from src.match import identify                           # noqa: E402
from src.plots import SNR_ORDER, plot_accuracy_lines, plot_heatmap  # noqa: E402
from src.utils import add_noise, load_audio_file         # noqa: E402

CLIP_LENGTHS = [3, 5, 10]
SNRS = [None, 20, 10, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="db/fingerprints.npz")
    ap.add_argument("--data", required=True, help="same folder used for build_db.py")
    ap.add_argument("--n", type=int, default=200, help="queries per condition")
    ap.add_argument("--song-seconds", type=float, default=30, help="usable length of each track")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()

    db = load_db(args.db)
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    rows = []

    for clip in CLIP_LENGTHS:
        # Same random queries for every SNR -> a fair, paired comparison.
        rng = np.random.default_rng(args.seed + clip)
        n = min(args.n, db.n_songs)
        song_ids = rng.choice(db.n_songs, size=n, replace=False)
        clips = []
        for sid in song_ids:
            offset = float(rng.uniform(0, max(args.song_seconds - clip - 0.5, 0)))
            y = load_audio_file(Path(args.data) / str(db.paths[sid]), duration=clip, offset=offset)
            clips.append((int(sid), y))

        for snr in SNRS:
            noise_rng = np.random.default_rng(args.seed + 1000 + clip)
            correct, scores = 0, []
            for sid, y in clips:
                res = identify(add_noise(y, snr, noise_rng), db)
                hit = res.matched and res.best["song_id"] == sid
                correct += hit
                if hit:
                    scores.append(res.best["score"])
            label = "clean" if snr is None else str(snr)
            row = {"clip_len_sec": clip, "snr_db": label, "n_queries": n,
                   "accuracy": round(correct / n, 4),
                   "mean_score_when_correct": round(float(np.mean(scores)), 1) if scores else 0}
            rows.append(row); print(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    plot_heatmap(df).savefig(out_dir / "heatmap.png", dpi=150)
    plot_accuracy_lines(df).savefig(out_dir / "accuracy_curves.png", dpi=150)

    # Markdown table to paste into the README
    snrs = [s for s in SNR_ORDER if s in set(df["snr_db"])]
    print("\n| Clip length | " + " | ".join(f"SNR {s}" + ("" if s == "clean" else " dB") for s in snrs) + " |")
    print("|---|" + "---|" * len(snrs))
    for clip in CLIP_LENGTHS:
        vals = [df[(df.clip_len_sec == clip) & (df.snr_db == s)]["accuracy"].iloc[0] for s in snrs]
        print(f"| {clip} s | " + " | ".join(f"{v:.0%}" for v in vals) + " |")
    print(f"\n(N = {df['n_queries'].iloc[0]} random queries per cell, seed = {args.seed})")


if __name__ == "__main__":
    main()
