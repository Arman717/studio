#!/usr/bin/env python3
"""Utility script that wraps GLASS training for screw images."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    from torchvision import transforms
except ModuleNotFoundError:
    print(
        "Missing dependencies. Please install PyTorch and torchvision as listed in GLASS requirements.",
        file=sys.stderr,
    )
    raise

from PIL import Image

try:
    import numpy as np
except ModuleNotFoundError:
    print("Missing dependency numpy.", file=sys.stderr)
    raise


def ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    subprocess.check_call(["git", "clone", "https://github.com/cqylunlun/GLASS", str(repo_dir)])


def _binary_dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=0)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    )


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=0)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )


def _binary_close(mask: np.ndarray) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask using flood fill."""
    padded = np.pad(mask, 1, constant_values=0)
    h, w = padded.shape
    stack = [(0, 0)]
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or padded[y, x] != 0:
            continue
        padded[y, x] = 2
        stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
    filled = padded != 2
    return filled[1:-1, 1:-1]



def segment_screw(img: Image.Image):
    """Segment the screw using Otsu thresholding while keeping dark anomalies."""
    np_img = np.array(img)
    gray = np_img.mean(axis=2).astype(np.uint8)
    hist = np.bincount(gray.flatten(), minlength=256)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    maximum = 0.0
    threshold = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            threshold = i
    mask = gray > threshold
    if mask.mean() > 0.5:
        mask = gray < threshold
    mask = _binary_close(mask)
    mask = _fill_holes(mask)
    rgb = np_img.copy()
    rgb[~mask] = 0
    return Image.fromarray(rgb), mask.astype(np.uint8)
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        # Determine the target square size by scanning all images and taking
        # the largest dimension. Images are never resized; shorter edges are
        # padded so the full screw remains in view.
        self.imagesize = 0
        for p in self.paths:
            with Image.open(p) as im:
                self.imagesize = max(self.imagesize, im.width, im.height)
        # Spatial resolution of the segmentation mask expected by the GLASS
        # discriminator. This is determined dynamically from the backbone's
        # patch shape and therefore populated later by ``set_mask_shape`` once
        # the model is instantiated.
        self.mask_shape: Optional[Tuple[int, int]] = None
        # Bypass GLASS distribution auto-detection, which can prematurely exit
        # without saving a checkpoint when the expected metadata file is
        # absent. Using the "manifold" identifier (value 2) lets training
        # proceed normally.
        self.distribution = 2
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.save_dir = Path(__file__).resolve().parent / "preprocessed_train"
        self.save_dir.mkdir(exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def set_mask_shape(self, shape: Tuple[int, int]) -> None:
        """Define the expected mask shape produced by the backbone."""
        self.mask_shape = shape

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        seg, mask = segment_screw(img)
        if mask.max() == 0:
            mask[0, 0] = 1

        # Crop tightly around the screw so no portion is discarded.
        ys, xs = np.where(mask)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        pad = 10
        xmin = max(0, xmin - pad)
        ymin = max(0, ymin - pad)
        xmax = min(seg.width - 1, xmax + pad)
        ymax = min(seg.height - 1, ymax + pad)
        seg = seg.crop((xmin, ymin, xmax + 1, ymax + 1))
        mask = mask[ymin : ymax + 1, xmin : xmax + 1]

        # Pad the cropped screw to the common square size without resizing.
        canvas = Image.new("RGB", (self.imagesize, self.imagesize))
        left = (self.imagesize - seg.width) // 2
        top = (self.imagesize - seg.height) // 2
        canvas.paste(seg, (left, top))
        seg = canvas
        mask_canvas = np.zeros((self.imagesize, self.imagesize), dtype=np.uint8)
        mask_canvas[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
        mask = mask_canvas

        out_path = self.save_dir / f"{self.paths[idx].stem}.png"
        seg.save(out_path)

        tensor = self.tf(seg)
        if self.mask_shape is None:
            raise RuntimeError("mask_shape is not set for ImageDataset")
        mask_img_full = Image.fromarray(mask * 255)
        mask_img = mask_img_full.resize(
            (self.mask_shape[1], self.mask_shape[0]), Image.NEAREST
        )
        mask_s = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0)
        if mask_s.max() == 0:
            mask_s[0, 0] = 1.0
        return {
            "image": tensor,
            "aug": tensor,
            # Segmentation mask downsampled to match the backbone feature map.
            # Providing a mask with at least one positive value prevents GLASS
            # from computing statistics on empty tensors during discriminator
            # training.
            "mask_s": mask_s,
            "is_anomaly": torch.tensor(0),
            "mask_gt": torch.zeros(1, self.imagesize, self.imagesize),
            "image_path": str(self.paths[idx]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GLASS model")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument(
        "--backbone",
        default="efficientnet_b5",
        help="GLASS backbone name (e.g. efficientnet_b5, wideresnet50)",
    )
    parser.add_argument("images", nargs="*", help="Training images")
    args = parser.parse_args()

    if len(args.images) == 0:
        print("No training images were provided", file=sys.stderr)
        sys.exit(1)

    repo_dir = Path(__file__).resolve().parent / "glass_repo"
    ensure_repo(repo_dir)
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    import glass as glass_mod  # type: ignore
    import backbones  # type: ignore
    import metrics as glass_metrics  # type: ignore

    # GLASS's evaluation metrics rely on ROC AUC scores, which require both
    # positive and negative labels. Our screw dataset contains only normal
    # samples, so evaluation would otherwise raise a ValueError. Override the
    # metric helpers to return zeros when the ground truth lacks class
    # diversity.

    _orig_img_metrics = glass_metrics.compute_imagewise_retrieval_metrics
    _orig_px_metrics = glass_metrics.compute_pixelwise_retrieval_metrics

    def _safe_img_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path="training"):
        if len(set(map(int, anomaly_ground_truth_labels))) <= 1:
            return {"auroc": 0.0, "ap": 0.0}
        return _orig_img_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path)

    def _safe_px_metrics(anomaly_segmentations, ground_truth_masks, path="train"):
        try:
            import numpy as np

            if isinstance(ground_truth_masks, list):
                gt = np.stack(ground_truth_masks)
            else:
                gt = np.asarray(ground_truth_masks)
            if np.unique(gt.astype(int)).size <= 1:
                return {"auroc": 0.0, "ap": 0.0}
        except Exception:
            pass
        return _orig_px_metrics(anomaly_segmentations, ground_truth_masks, path)

    glass_metrics.compute_imagewise_retrieval_metrics = _safe_img_metrics
    glass_metrics.compute_pixelwise_retrieval_metrics = _safe_px_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageDataset(args.images)

    backbone = backbones.load(args.backbone)
    model = glass_mod.GLASS(device)
    # Hyperparameters align with the optimal configuration suggested in the
    # GLASS paper for unsupervised anomaly detection.
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, dataset.imagesize, dataset.imagesize),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        meta_epochs=640,
        eval_epochs=1,
        dsc_layers=2,
        dsc_hidden=1024,
        pre_proj=1,
        mining=1,
        noise=0.015,
        radius=0.75,
        p=0.5,
        step=20,
        limit=392,
    )

    # Determine the discriminator's expected mask resolution from the backbone's
    # patch shape and update the dataset accordingly.
    dummy = torch.zeros(1, 3, dataset.imagesize, dataset.imagesize).to(device)
    _, patch_shapes = model._embed(dummy, evaluation=True, provide_patch_shapes=True)
    dataset.set_mask_shape(patch_shapes[0])

    # Use batch size 8 as recommended in the GLASS paper.
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    dataloaders = {"training": loader, "testing": loader}

    model.set_model_dir(tempfile.mkdtemp(), "custom")
    try:
        model.trainer(dataloaders["training"], dataloaders["testing"], "custom")
    except KeyboardInterrupt:
        pass

    ckpt = Path(model.ckpt_dir) / "ckpt.pth"
    if not ckpt.exists():
        print("Training stopped before producing a checkpoint", file=sys.stderr)
        sys.exit(1)
    shutil.copy(ckpt, args.output)

    print(json.dumps({"modelId": str(args.output)}))


if __name__ == "__main__":
    main()
