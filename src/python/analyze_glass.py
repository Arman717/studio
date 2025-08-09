#!/usr/bin/env python3
"""Analyze screw images using SAM-based background removal."""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def remove_background(img: np.ndarray, checkpoint: str) -> Tuple[np.ndarray, np.ndarray]:
    """Remove background from an image using SAM.

    Args:
        img: Input image as a numpy array in BGR or RGB format.
        checkpoint: Path to SAM checkpoint file.

    Returns:
        The masked image and boolean mask.
    """
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(rgb)
    if not masks:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
    else:
        best = max(masks, key=lambda m: m.get("area", 0))
        mask = best["segmentation"].astype(bool)

    result = img.copy()
    result[~mask] = 0
    return result, mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze screw image (placeholder)")
    default_ckpt = Path(__file__).resolve().parent / "sam" / "sam_vit_b.pth"
    parser.add_argument(
        "--sam-checkpoint",
        default=str(default_ckpt),
        help="Path to SAM checkpoint",
    )
    parser.add_argument("image", help="Image to analyze")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    masked, mask = remove_background(image, args.sam_checkpoint)
    cv2.imwrite("masked.png", masked)
    cv2.imwrite("mask.png", mask.astype("uint8") * 255)
    print("Background removed using SAM")


if __name__ == "__main__":
    main()
