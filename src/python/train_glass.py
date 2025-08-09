#!/usr/bin/env python3
"""Training utilities for glass inspection with SAM-based background removal."""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def remove_background(img: np.ndarray, checkpoint: str) -> Tuple[np.ndarray, np.ndarray]:
    """Remove background from an RGB/BGR image using the Segment Anything Model.

    Args:
        img: Input image as a numpy array in BGR or RGB format.
        checkpoint: Path to the SAM checkpoint file.

    Returns:
        A tuple of (masked_image, mask) where mask is a boolean array.
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
    parser = argparse.ArgumentParser(description="Train glass model (placeholder)")
    default_ckpt = Path(__file__).resolve().parent / "sam" / "sam_vit_b.pth"
    parser.add_argument(
        "--sam-checkpoint",
        default=str(default_ckpt),
        help="Path to SAM checkpoint",
    )
    parser.add_argument("image", help="Sample image for background removal")
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
