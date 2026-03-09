#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generate caption .txt files for images (SD1.5 UNet LoRA training).
- Scans a folder for .jpg/.jpeg/.png
- Creates a same-named .txt with a randomly chosen flame caption
- Skips existing .txt unless --overwrite

Usage:
  python auto_make_caption.py --data ./data/fire_pairs
"""
import argparse
import random
import sys
from pathlib import Path

# ---------- Caption Pools (ENGLISH ONLY) ----------
CAPTIONS_EN = {
    "basic": [
        "a photo of realistic orange and yellow flames",
        "realistic fire flames, natural orange and red colors",
        "bright realistic fire with glowing embers",
        "a close-up of burning fire texture, glowing hot",
        "realistic flame texture, orange and yellow gradient",
    ],
    "bright": [
        "bright glowing fire, intense orange light, realistic heat shimmer",
        "flames with high temperature glow, vivid orange and yellow colors",
        "cinematic shot of bright fire flames, glowing sparks",
        "realistic fire texture, high exposure, radiant light",
    ],
    "warm": [
        "soft warm flame, orange light, gentle glow, cinematic lighting",
        "warm campfire flame texture, realistic and detailed",
        "soft glowing fire with smooth gradient, natural light",
    ],
    "dynamic": [
        "intense roaring flame, dynamic motion, heat distortion",
        "fierce burning fire, glowing core, dramatic lighting",
        "blazing orange and red flames, strong hot fire",
        "wildfire-like flames, powerful and vivid",
    ],
    "detail": [
        "macro shot of fire flame details, orange gradients and glowing core",
        "realistic flame texture pattern, burning edges and light bloom",
        "glowing fire sparks, high contrast flame, realistic color depth",
    ],
}

EXTS = {".jpg", ".jpeg", ".png"}

def build_pool(weights: str):
    """Build a flat list of captions according to weights."""
    pools = CAPTIONS_EN

    default = {"basic": 5, "bright": 3, "warm": 2, "dynamic": 3, "detail": 2}
    w = {}

    if weights:
        try:
            for kv in weights.split(","):
                k, v = kv.split("=")
                w[k.strip()] = int(v.strip())
        except Exception:
            print("[warn] Bad weights string, using defaults.", file=sys.stderr)
            w = default

    if not w:
        w = default

    bag = []
    for k, cands in pools.items():
        times = max(0, w.get(k, 0))
        for _ in range(times):
            bag.extend(cands)

    if not bag:
        for cands in pools.values():
            bag.extend(cands)

    return bag


def main():
    ap = argparse.ArgumentParser(description="Auto caption generator for flame LoRA dataset")
    ap.add_argument("--data", type=str, required=True, help="Folder containing images")
    ap.add_argument("--weights", type=str, default="basic=5,bright=3,warm=2,dynamic=3,detail=2",
                    help="Category weights")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    ap.add_argument("--suffix", type=str, default="", help="Optional suffix (e.g., ', realistic lighting')")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.data)

    if not root.exists():
        print(f"[error] Data folder not found: {root}")
        sys.exit(1)

    pool = build_pool(args.weights)

    images = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
    if not images:
        print(f"[error] No images found in {root}")
        sys.exit(1)

    made, skipped = 0, 0

    for img in images:
        txt = img.with_suffix(".txt")

        if txt.exists() and not args.overwrite:
            skipped += 1
            continue

        cap = random.choice(pool)

        if args.suffix:
            cap = f"{cap}{args.suffix}"

        try:
            txt.write_text(cap + "\n", encoding="utf-8")
            made += 1
        except Exception as e:
            print(f"[warn] Failed to write {txt.name}: {e}", file=sys.stderr)

    print(f"[done] images={len(images)}, created={made}, skipped={skipped}")

if __name__ == "__main__":
    main()