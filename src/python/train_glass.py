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

try:
    import numpy as np
except ModuleNotFoundError:
    print("Missing dependency numpy.", file=sys.stderr)
    raise

SAM2_AVAILABLE = False
try:  # Prefer Segment Anything 2 if available
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
    SAM2_AVAILABLE = True
except ModuleNotFoundError:
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # type: ignore
    except ModuleNotFoundError:
        print("Missing dependency sam2 or segment-anything.", file=sys.stderr)
        raise


def ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    subprocess.check_call(["git", "clone", "https://github.com/cqylunlun/GLASS", str(repo_dir)])


_sam_generator = None


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



def segment_with_sam(img: Image.Image):
    """Use Segment Anything (v2 if installed) to isolate the screw."""
    global _sam_generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _sam_generator is None:
        if SAM2_AVAILABLE:
            checkpoint = Path(__file__).resolve().parent / "sam2.1_hiera_large.pt"
            cfg = Path(__file__).resolve().parent / "sam2.1_hiera_l.yaml"
            if not checkpoint.exists() or not cfg.exists():
                import urllib.request
                if not checkpoint.exists():
                    url_ckpt = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
                    urllib.request.urlretrieve(url_ckpt, checkpoint)
                if not cfg.exists():
                    url_cfg = "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
                    urllib.request.urlretrieve(url_cfg, cfg)
            sam = build_sam2(str(cfg), str(checkpoint), device=device, apply_postprocessing=False)
            params = dict(
                points_per_side=64,
                points_per_batch=256,
                crop_n_layers=1,
                crop_overlap_ratio=0.5,
                pred_iou_thresh=0.82,
                stability_score_thresh=0.88,
                min_mask_region_area=100,
                output_mode="binary_mask",
            )
            _sam_generator = SAM2AutomaticMaskGenerator(sam, **params)
        else:
            checkpoint = Path(__file__).resolve().parent / "sam_vit_h_4b8939.pth"
            if not checkpoint.exists():
                import urllib.request
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                urllib.request.urlretrieve(url, checkpoint)
            sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
            sam.to(device)
            params = dict(
                points_per_side=64,
                points_per_batch=256,
                crop_n_layers=1,
                crop_overlap_ratio=0.5,
                pred_iou_thresh=0.82,
                stability_score_thresh=0.88,
                min_mask_region_area=100,
                output_mode="binary_mask",
            )
            _sam_generator = SamAutomaticMaskGenerator(sam, **params)
    np_img = np.array(img)
    gray = np_img.mean(axis=2).astype(np.uint8)
    coords = np.column_stack(np.where(gray > 0))
    h, w = gray.shape
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        pad = 20
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(h, y1 + pad)
        x1 = min(w, x1 + pad)
    else:
        y0, x0, y1, x1 = 0, 0, h, w
    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    scale = 1.0
    if min(cw, ch) < 80:
        scale = 80.0 / min(cw, ch)
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.BICUBIC)
    np_crop = np.array(crop)
    masks = _sam_generator.generate(np_crop)
    if not masks:
        mask = np.ones(np_crop.shape[:2], dtype=bool)
    else:
        total_area = np_crop.shape[0] * np_crop.shape[1]
        valid = [m for m in masks if m.get("area", 0) < 0.9 * total_area]
        valid.sort(key=lambda m: m.get("area", 0), reverse=True)
        def bbox(m):
            ys, xs = np.where(m)
            return ys.min(), xs.min(), ys.max(), xs.max()
        base = valid[0]["segmentation"]
        base_box = bbox(base)
        for cand in valid[1:]:
            seg = cand["segmentation"]
            ys, xs = np.where(seg)
            if ys.size == 0:
                continue
            y0c, x0c, y1c, x1c = ys.min(), xs.min(), ys.max(), xs.max()
            elong = max(y1c - y0c + 1, x1c - x0c + 1) / max(1, min(y1c - y0c + 1, x1c - x0c + 1))
            bb0 = max(0, base_box[0] - 2)
            bb1 = max(0, base_box[1] - 2)
            bb2 = min(np_crop.shape[0] - 1, base_box[2] + 2)
            bb3 = min(np_crop.shape[1] - 1, base_box[3] + 2)
            if (y0c <= bb2 and y1c >= bb0 and x0c <= bb3 and x1c >= bb1) or elong >= 2.5:
                base |= seg
                base_box = bbox(base)
        mask = _binary_close(base)
    if scale != 1.0:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_img = mask_img.resize((cw, ch), Image.NEAREST)
        mask = np.array(mask_img, dtype=bool)
    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y0:y1, x0:x1] = mask
    rgb = np_img.copy()
    rgb[~full_mask] = 0
    return Image.fromarray(rgb), full_mask.astype(np.uint8)
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

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        seg, mask = segment_with_sam(img)
        seg = seg.resize((self.imagesize, self.imagesize))
        out_path = self.save_dir / f"{self.paths[idx].stem}.png"
        seg.save(out_path)
        tensor = self.tf(seg)
        mask_size = self.imagesize // self.downsampling
        mask_img_full = Image.fromarray(mask * 255).resize((self.imagesize, self.imagesize), Image.NEAREST)
        mask_img = mask_img_full.resize((mask_size, mask_size), Image.NEAREST)
        mask_s = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0)
        return {
            "image": tensor,
            "aug": tensor,
            # Segmentation mask downsampled to match the 36×36 feature map.
            # Providing a mask with at least one positive value prevents
            # GLASS from computing statistics on empty tensors during
            # discriminator training.
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
    # Use batch size 8 as recommended in the GLASS paper.
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    dataloaders = {"training": loader, "testing": loader}

    backbone = backbones.load("wideresnet50")
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

    model.set_model_dir(tempfile.mkdtemp(), "custom")
    model.trainer(dataloaders["training"], dataloaders["testing"], "custom")

    ckpt = Path(model.ckpt_dir) / "ckpt.pth"
    shutil.copy(ckpt, args.output)

    print(json.dumps({"modelId": str(args.output)}))


if __name__ == "__main__":
    main()
