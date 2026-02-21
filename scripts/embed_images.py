#!/usr/bin/env python3
"""Pre-compute CLIP ViT-B/32 embeddings for Tiny ImageNet.

Fallback for when fastembed-rs can't compile (ONNX Runtime build issues).
Outputs binary file loadable by vsaclip::ingest::load_embeddings().

Format: [u32 count] [u32 dim] [f32 × count × dim]  (little-endian)

Usage:
    python3 scripts/embed_images.py [--data-dir data] [--output data/embeddings.bin] [--batch-size 64]
"""

import argparse
import os
import struct
import sys
from pathlib import Path


def scan_images(data_dir: Path) -> list[tuple[str, str]]:
    """Scan Tiny ImageNet train directory. Returns [(path, class_id), ...]."""
    train_dir = data_dir / "tiny-imagenet-200" / "train"
    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found. Download Tiny ImageNet first.", file=sys.stderr)
        sys.exit(1)

    entries = []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        images_dir = class_dir / "images"
        if not images_dir.exists():
            continue
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() in (".jpeg", ".jpg"):
                entries.append((str(img_path), class_id))

    return entries


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP embeddings for Tiny ImageNet")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/embeddings.bin"))
    parser.add_argument("--labels-output", type=Path, default=Path("data/labels.txt"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("Scanning images...")
    entries = scan_images(args.data_dir)
    print(f"Found {len(entries)} images in {len(set(c for _, c in entries))} classes")

    if len(entries) == 0:
        print("No images found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Save labels
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels_output, "w") as f:
        for path, class_id in entries:
            f.write(f"{class_id}\t{path}\n")
    print(f"Labels saved to {args.labels_output}")

    # Load CLIP model
    print("Loading CLIP ViT-B/32 via fastembed...")
    from fastembed import ImageEmbedding

    model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    dim = 512

    # Embed in batches
    all_embeddings = []
    paths = [p for p, _ in entries]
    total = len(paths)
    bs = args.batch_size

    for start in range(0, total, bs):
        end = min(start + bs, total)
        batch_paths = paths[start:end]
        batch_embs = list(model.embed(batch_paths))
        all_embeddings.extend(batch_embs)
        print(f"  Embedded {end}/{total} ({100*end//total}%)", end="\r")

    print(f"\nEmbedded {len(all_embeddings)} images (dim={dim})")

    # Save as binary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(struct.pack("<II", len(all_embeddings), dim))
        for emb in all_embeddings:
            for val in emb:
                f.write(struct.pack("<f", float(val)))

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
