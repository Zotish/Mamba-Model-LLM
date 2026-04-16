"""
download_tinystories.py — Download a subset of TinyStories dataset.

TinyStories (Microsoft Research, 2023):
  - Short stories written in simple English
  - Designed for training small language models
  - Full dataset: ~2GB  |  We download a manageable subset

Usage:
    python download_tinystories.py              # default: 50K stories (~30MB)
    python download_tinystories.py --n 10000    # smaller: 10K stories
    python download_tinystories.py --n 100000   # larger:  100K stories
"""

import os
import argparse

BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
OUT_PATH  = os.path.join(BASE_DIR, "data", "input.txt")
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)


def download(n_stories: int):
    print(f"Downloading TinyStories — {n_stories:,} stories from HuggingFace...")
    print("(This may take a few minutes on first run — HF caches locally after)")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets  first")
        raise

    # stream=True lets us take just the first N without downloading everything
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    stories = []
    for i, ex in enumerate(ds):
        if i >= n_stories:
            break
        text = ex.get("text", "").strip()
        if text:
            stories.append(text)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,} / {n_stories:,} stories downloaded...")

    print(f"\n✅ Got {len(stories):,} stories")

    # Write to input.txt — stories separated by newline
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stories))

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    chars   = sum(len(s) for s in stories)
    print(f"✅ Saved → {OUT_PATH}")
    print(f"   File size  : {size_mb:.1f} MB")
    print(f"   Characters : {chars:,}")
    print(f"   Stories    : {len(stories):,}")
    print(f"\nNow train with:")
    print(f"   python train_mamba.py --mps --run_hours 3 --block_size 256")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50_000, help="number of stories to download")
    args = p.parse_args()
    download(args.n)
