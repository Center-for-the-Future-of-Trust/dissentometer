#!/usr/bin/env python3
import os, sys, argparse
from pathlib import Path
from typing import List, Tuple

def find_files(root: Path, pattern: str) -> List[Path]:
    # Recursive, deterministic order
    return sorted(root.rglob(pattern))

def even_slices(n: int, k: int) -> List[Tuple[int, int]]:
    """Return k (start, end) index pairs that partition range(n) contiguously and evenly.
    First (n % k) slices get 1 extra item. end is exclusive.
    Works even if k > n (some shards will be empty)."""
    if k <= 0:
        raise ValueError("num-shards must be >= 1")
    base = n // k
    rem = n % k
    slices: List[Tuple[int, int]] = []
    start = 0
    for i in range(k):
        length = base + (1 if i < rem else 0)
        end = start + length
        slices.append((start, end))
        start = end
    return slices

def write_buckets(files: List[Path], buckets_dir: Path, bucket_size: int) -> int:
    buckets_dir.mkdir(parents=True, exist_ok=True)
    # clear old lists
    for p in buckets_dir.glob("bucket_*.lst"):
        p.unlink()
    b = 0
    for i in range(0, len(files), bucket_size):
        out = buckets_dir / f"bucket_{b:03d}.lst"
        with out.open("w", encoding="utf-8") as w:
            for fp in files[i:i+bucket_size]:
                w.write(str(fp.resolve())+"\n")
        b += 1
    return b

def link_shard(files: List[Path], inputs_dir: Path) -> int:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    # clear existing links/files in inputs_dir
    for p in inputs_dir.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()
    for fp in files:
        (inputs_dir / fp.name).symlink_to(fp.resolve())
    return len(files)

def main():
    ap = argparse.ArgumentParser(description="Split files into N shards (default 3) and bucketize each shard.")
    ap.add_argument("--data-root", required=True, help="Directory to search recursively for files.")
    ap.add_argument("--file-glob", default="*.csv", help="Glob of files to include (default: *.csv)." )
    ap.add_argument("--out-base", required=True, help="Base directory where shard folders will be created." )
    ap.add_argument("--bucket-size", type=int, default=300, help="Files per bucket list (default: 300)." )
    ap.add_argument("--num-shards", type=int, default=3, help="Number of shards to create (default: 3)." )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_base = Path(args.out_base)

    files = find_files(data_root, args.file_glob)
    if not files:
        print(f"[WARN] No files matched {args.file_glob} under {data_root}", file=sys.stderr)
        return 0

    n = len(files)
    k = max(1, args.num_shards)
    slices = even_slices(n, k)

    print(f"[INFO] total={n} shards={k} bucket_size={args.bucket_size}")
    for i, (start, end) in enumerate(slices, start=1):
        shard_files = files[start:end]
        shard_dir = out_base / f"shard{i}"
        n_linked = link_shard(shard_files, shard_dir / "inputs")
        n_buckets = write_buckets(shard_files, shard_dir / "buckets", args.bucket_size)
        print(f"[OK] shard{i}: files={n_linked} buckets={n_buckets} -> {shard_dir}")

    print(f"[DONE] Output at: {out_base}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
