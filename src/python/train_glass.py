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


def ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    subprocess.check_call(["git", "clone", "https://github.com/cqylunlun/GLASS", str(repo_dir)])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.imagesize = 288
        # GLASS training expects a "mask_s" tensor whose spatial resolution
        # matches the backbone's feature map (image size divided by the
        # dataset downsampling factor). The original GLASS datasets use a
        # downsampling factor of 8, yielding a 36×36 mask for 288×288 inputs.
        # Without this, the discriminator receives a mismatched mask and
        # raises an IndexError when indexing patch embeddings.
        self.downsampling = 8
        # Bypass GLASS distribution auto-detection, which can
        # prematurely exit without saving a checkpoint when the
        # expected metadata file is absent. Using the "manifold"
        # identifier (value 2) lets training proceed normally.
        self.distribution = 2
        self.tf = transforms.Compose(
            [
                transforms.Resize((self.imagesize, self.imagesize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.tf(img)
        mask_size = self.imagesize // self.downsampling
        mask_s = torch.ones(mask_size, mask_size)
        return {
            "image": tensor,
            "aug": tensor,
            # Placeholder mask required by GLASS. Using the correct spatial
            # dimensions (36×36) keeps mask indices aligned with extracted
            # features during discriminator training. The mask must contain at
            # least one positive value; otherwise, GLASS computes statistics on
            # an empty tensor and raises a RuntimeError. Filling the mask with
            # ones satisfies this requirement without relying on ground-truth
            # annotations.
            "mask_s": mask_s,
            "is_anomaly": torch.tensor(0),
            "mask_gt": torch.zeros(1, self.imagesize, self.imagesize),
            "image_path": str(self.paths[idx]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GLASS model")
    parser.add_argument("--output", required=True, help="Output model path")
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    dataloaders = {"training": loader, "testing": loader}

    backbone = backbones.load("wideresnet50")
    model = glass_mod.GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, dataset.imagesize, dataset.imagesize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        meta_epochs=1,
        eval_epochs=1,
    )

    model.set_model_dir(tempfile.mkdtemp(), "custom")
    model.trainer(dataloaders["training"], dataloaders["testing"], "custom")

    ckpt = Path(model.ckpt_dir) / "ckpt.pth"
    shutil.copy(ckpt, args.output)

    print(json.dumps({"modelId": str(args.output)}))


if __name__ == "__main__":
    main()
