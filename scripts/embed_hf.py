#!/usr/bin/env python3
"""Pre-compute CLIP ViT-B/32 embeddings from HuggingFace Tiny ImageNet dataset.

Alternative to embed_images.py when direct download from cs231n.stanford.edu fails.
Uses `datasets` library to load images and `fastembed` for CLIP embeddings.

Output format (loadable by vsaclip::ingest::load_embeddings()):
  [u32 count] [u32 dim] [f32 × count × dim]  (little-endian)

Usage:
    python3 scripts/embed_hf.py [--output data/embeddings.bin] [--batch-size 64]
"""

import argparse
import struct
import sys
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP embeddings from HF Tiny ImageNet")
    parser.add_argument("--output", type=Path, default=Path("data/embeddings.bin"))
    parser.add_argument("--labels-output", type=Path, default=Path("data/labels.txt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0=all)")
    args = parser.parse_args()

    # Load dataset
    print("Loading Tiny ImageNet from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    class_names = ds.features["label"].names
    n = len(ds) if args.max_images <= 0 else min(args.max_images, len(ds))
    print(f"  {n} images, {len(class_names)} classes")

    # Load CLIP model
    print("Loading CLIP ViT-B/32 via fastembed...")
    from fastembed import ImageEmbedding
    model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    dim = 512

    # Process in batches
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_labels = []
    bs = args.batch_size

    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = ds.select(range(start, end))

        # Save images to temp files (fastembed expects file paths)
        tmp_dir = tempfile.mkdtemp(prefix="vsaclip_")
        tmp_paths = []
        for i, example in enumerate(batch):
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            path = os.path.join(tmp_dir, f"img_{i}.jpg")
            img.save(path, "JPEG")
            tmp_paths.append(path)
            all_labels.append(class_names[example["label"]])

        # Embed batch
        try:
            batch_embs = list(model.embed(tmp_paths))
            all_embeddings.extend(batch_embs)
        except Exception as e:
            print(f"\n  WARNING: batch {start}-{end} failed: {e}")
            # Fill with zeros for failed batch
            for _ in range(end - start):
                all_embeddings.append([0.0] * dim)

        # Cleanup temp files
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        pct = end * 100 // n
        print(f"  Embedded {end}/{n} ({pct}%)", end="\r")

    print(f"\n  Done: {len(all_embeddings)} embeddings (dim={dim})")

    # Save labels
    with open(args.labels_output, "w") as f:
        for label in all_labels:
            f.write(f"{label}\tHF\n")
    print(f"  Labels saved to {args.labels_output}")

    # Save embeddings as binary
    with open(args.output, "wb") as f:
        f.write(struct.pack("<II", len(all_embeddings), dim))
        for emb in all_embeddings:
            for val in emb:
                f.write(struct.pack("<f", float(val)))

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"  Embeddings saved to {args.output} ({size_mb:.1f} MB)")

    print("\nNext step: cargo run --release --bin poc")


if __name__ == "__main__":
    main()
