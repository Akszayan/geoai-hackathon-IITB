# scripts/make_val_from_train.py
"""
Pick N random items from train_tiles.txt and move them to val_tiles.txt (or copy).
Usage:
    python -u scripts/make_val_from_train.py --n 200 --seed 42 --copy
"""
import argparse, random, shutil
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=str, default="data/meta/train_tiles.txt")
    p.add_argument("--val", type=str, default="data/meta/val_tiles.txt")
    p.add_argument("--copy", action="store_true", help="Copy to val (keep train); default behaviour removes from train")
    args = p.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    assert train_path.exists(), f"Train list not found: {train_path}"

    lines = [l.strip() for l in open(train_path, "r", encoding="utf8") if l.strip()]
    random.Random(args.seed).shuffle(lines)
    n = min(args.n, len(lines))
    picked = lines[:n]
    if args.copy:
        # append to val
        existing = []
        if val_path.exists():
            existing = [l.strip() for l in open(val_path, "r", encoding="utf8") if l.strip()]
        new_val = existing + picked
        open(val_path, "w", encoding="utf8").write("\n".join(new_val))
        print(f"[COPY] Wrote {n} tiles to {val_path} (kept train unchanged).")
    else:
        # remove picked from train, write both
        remaining = lines[n:]
        open(train_path, "w", encoding="utf8").write("\n".join(remaining))
        open(val_path, "w", encoding="utf8").write("\n".join(picked))
        print(f"[MOVE] Moved {n} tiles to {val_path}. Train count now {len(remaining)}.")

if __name__ == "__main__":
    main()
