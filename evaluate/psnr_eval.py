#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG")


def list_images(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.endswith(IMG_EXTS)]
    files = sorted(files)
    return [os.path.join(folder, f) for f in files]


def psnr_numpy(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    img1 img2 H x W x C uint8 or float32 range 0 255
    """
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_val ** 2) / mse)


def masked_psnr_numpy(
    img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, max_val: float = 255.0
) -> float:
    """
    Calculate PSNR in mask 1 region
    img1 img2 H x W x C
    mask H x W 0 1 or 0 255 grayscale image
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    # Normalize to 0 1
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask_bin = (mask > 0.5).astype(np.float32)

    # Avoid empty mask
    valid = mask_bin.sum()
    if valid < 1:
        return float("nan")

    diff = (img1 - img2) * mask_bin[..., None]
    mse = (diff ** 2).sum() / (valid * img1.shape[2])
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_val ** 2) / mse)


def build_name_map(paths: List[str]) -> dict:
    """
    paths list of full paths
    return basename to full path
    """
    m = {}
    for p in paths:
        name = os.path.basename(p)
        m[name] = p
    return m


def compute_psnr_for_dirs(
    real_dir: str, fake_dir: str, mask_dir: str = None, resize_to_real: bool = True
) -> Tuple[List[float], List[float]]:
    real_paths = list_images(real_dir)
    fake_paths = list_images(fake_dir)

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise ValueError("No images found in real dir or fake dir")

    real_map = build_name_map(real_paths)
    fake_map = build_name_map(fake_paths)

    mask_map = {}
    if mask_dir:
        mask_paths = list_images(mask_dir)
        mask_map = build_name_map(mask_paths)

    psnr_vals = []
    psnr_mask_vals = []

    print(f"info real images {len(real_paths)}")
    print(f"info fake images {len(fake_paths)}")
    if mask_dir:
        print(f"info mask images {len(mask_map)}")

    common_names = sorted(set(real_map.keys()) & set(fake_map.keys()))
    print(f"info matched by filename {len(common_names)}")

    for name in tqdm(common_names, desc="Computing PSNR"):
        real_p = real_map[name]
        fake_p = fake_map[name]

        try:
            real_img = Image.open(real_p).convert("RGB")
            fake_img = Image.open(fake_p).convert("RGB")

            if resize_to_real and fake_img.size != real_img.size:
                fake_img = fake_img.resize(real_img.size, Image.BILINEAR)

            real_np = np.array(real_img)
            fake_np = np.array(fake_img)

            ps = psnr_numpy(real_np, fake_np, max_val=255.0)
            psnr_vals.append(ps)

            if mask_dir and name in mask_map:
                mask_p = mask_map[name]
                mask_img = Image.open(mask_p).convert("L")
                if mask_img.size != real_img.size:
                    mask_img = mask_img.resize(real_img.size, Image.NEAREST)
                mask_np = np.array(mask_img)
                ps_m = masked_psnr_numpy(real_np, fake_np, mask_np, max_val=255.0)
                psnr_mask_vals.append(ps_m)

        except Exception as e:
            print(f"warn Skipping {name} {e}")
            continue

    return psnr_vals, psnr_mask_vals


def main():
    ap = argparse.ArgumentParser(description="Compute PSNR between real and fake image folders")
    ap.add_argument("--real-dir", type=str, required=True, help="Real image folder reference")
    ap.add_argument("--fake-dir", type=str, required=True, help="Generated image folder comparison")
    ap.add_argument("--mask-dir", type=str, default=None, help="Optional mask folder used for Masked PSNR")
    ap.add_argument("--no-resize", action="store_true", help="Do not resize fake image to real image size")
    args = ap.parse_args()

    print("PSNR evaluation script")
    print(f"real dir {args.real_dir}")
    print(f"fake dir {args.fake_dir}")
    if args.mask_dir:
        print(f"mask dir {args.mask_dir}")
    print("====================")

    psnr_vals, psnr_mask_vals = compute_psnr_for_dirs(
        args.real_dir,
        args.fake_dir,
        mask_dir=args.mask_dir,
        resize_to_real=not args.no_resize,
    )

    if len(psnr_vals) == 0:
        print("Failed to calculate any PSNR please check if filenames match")
        return

    mean_psnr = float(np.mean(psnr_vals))
    std_psnr = float(np.std(psnr_vals))
    print("\nResult Full image PSNR")
    print(f"  Sample count {len(psnr_vals)}")
    print(f"  Average {mean_psnr:.4f} dB")
    print(f"  Standard deviation {std_psnr:.4f} dB")

    if psnr_mask_vals:
        psnr_mask_clean = [v for v in psnr_mask_vals if not np.isnan(v)]
        if psnr_mask_clean:
            mean_psnr_m = float(np.mean(psnr_mask_clean))
            std_psnr_m = float(np.std(psnr_mask_clean))
            print("\nResult Masked PSNR only in mask 1 region")
            print(f"  Sample count {len(psnr_mask_clean)}")
            print(f"  Average {mean_psnr_m:.4f} dB")
            print(f"  Standard deviation {std_psnr_m:.4f} dB")

    print("\nDone")


if __name__ == "__main__":
    main()