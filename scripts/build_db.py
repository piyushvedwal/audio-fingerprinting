"""Build the compact fingerprint database OFFLINE (on your laptop).

Usage:
    python scripts/build_db.py --data data/genres_original --out db/fingerprints.npz
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.database import build_from_fingerprints, save_db      # noqa: E402
from src.fingerprint import SR, fingerprint                    # noqa: E402
from src.utils import AUDIO_EXTS, load_audio_file              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="folder with audio files (searched recursively)")
    ap.add_argument("--out", default="db/fingerprints.npz")
    ap.add_argument("--max-seconds", type=float, default=60,
                    help="only fingerprint the first N seconds of each track")
    args = ap.parse_args()

    root = Path(args.data)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"No audio files found in {root}")

    fps, names, genres, paths, skipped = [], [], [], [], []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        try:
            y = load_audio_file(p, duration=args.max_seconds)
            if len(y) < SR:                       # < 1 second: useless / corrupt
                raise ValueError("audio too short")
            hashes, times = fingerprint(y)
            if len(hashes) == 0:
                raise ValueError("no fingerprints produced")
        except Exception as exc:                  # e.g. GTZAN jazz.00054.wav is corrupt
            skipped.append(f"{p}  ->  {exc}")
            print(f"[{i}/{len(files)}] SKIP {p.name}: {exc}")
            continue
        fps.append((hashes, times))
        names.append(p.stem)
        genres.append(p.parent.name if p.parent != root else "")
        paths.append(p.relative_to(root).as_posix())
        print(f"[{i}/{len(files)}] {p.name}: {len(hashes)} hashes")

    db = build_from_fingerprints(fps, names, genres, paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_db(db, str(out))
    if skipped:
        log = out.with_suffix(".skipped.log")
        log.write_text("\n".join(skipped), encoding="utf-8")
        print(f"\nSkipped {len(skipped)} file(s), logged to {log}")
    print(f"\nDone: {db.n_songs} songs, {db.n_hashes:,} hashes, "
          f"{out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
