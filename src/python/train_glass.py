#!/usr/bin/env python3
"""Utility script that wraps GLASS training for screw images."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import os
import io
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


def _open_image(path: str) -> Image.Image:
    """Open an image from local disk or Google Cloud Storage."""
    if path.startswith("gs://"):
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        data = bucket.blob(blob_name).download_as_bytes()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(path).convert("RGB")


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


def _binary_open(mask: np.ndarray) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    from collections import deque

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        q.append((0, x))
        q.append((h - 1, x))
    for y in range(h):
        q.append((y, 0))
        q.append((y, w - 1))
    while q:
        y, x = q.popleft()
        if 0 <= y < h and 0 <= x < w and not visited[y, x] and mask[y, x] == 0:
            visited[y, x] = True
            q.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
    holes = (~visited) & (mask == 0)
    filled = mask.copy()
    filled[holes] = 1
    return filled



def segment_screw(
    img: Image.Image,
    background: Optional[Image.Image] = None,
    output_size: Optional[int] = None,
    threshold: int = 20,
):
    """Segment the screw by subtracting a background model and thresholding.

    ``background`` is a median-composited image of the empty rig. ``output_size``
    optionally resizes the full image and mask to this size. Returns the
    segmented RGB image and the binary mask.
    """
    np_img = np.array(img)
    if background is not None:
        bg = background.resize(img.size)
        np_bg = np.array(bg)
        diff = np.abs(np_img.astype(np.int16) - np_bg.astype(np.int16))
        gray = diff.mean(axis=2).astype(np.uint8)
    else:
        gray = np_img.mean(axis=2).astype(np.uint8)
    mask = gray > threshold
    mask = _binary_close(_binary_open(mask))
    mask = _fill_holes(mask)
    for _ in range(2):
        mask = _binary_dilate(mask)
    rgb = np_img.copy()
    rgb[~mask] = 0
    out_img = Image.fromarray(rgb)
    if output_size is not None:
        out_img = out_img.resize((output_size, output_size), Image.BILINEAR)
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (output_size, output_size), Image.NEAREST
        )
        mask = np.array(mask_img, dtype=np.uint8) // 255
    return out_img, mask.astype(np.uint8)
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, background: Optional[list[str]] = None):
        self.paths = [Path(p) for p in paths]
        first_img = _open_image(paths[0])
        # Use the smaller dimension of the first image to determine the
        # square resolution expected by the network, but cap it to 1024px to
        # avoid excessive memory usage that can terminate the training process
        # on modest hardware.
        self.imagesize = min(first_img.width, first_img.height, 1024)
        first_img.close()
        self.background: Optional[Image.Image] = None
        if background:
            bg_arrays = []
            for p in background:
                bg_img = _open_image(p)
                if bg_img.width < self.imagesize or bg_img.height < self.imagesize:
                    raise ValueError(
                        f"Background image must be at least {self.imagesize}px in both dimensions"
                    )
                bg_resized = bg_img.resize((self.imagesize, self.imagesize), Image.BILINEAR)
                bg_arrays.append(np.array(bg_resized, dtype=np.uint8))
            median = np.median(np.stack(bg_arrays, axis=0), axis=0).astype(np.uint8)
            self.background = Image.fromarray(median)
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
        img = _open_image(str(self.paths[idx]))
        if img.width < self.imagesize or img.height < self.imagesize:
            raise ValueError(
                f"Training images must be at least {self.imagesize}px in both dimensions"
            )
        # Resize to the common square size determined from the first image
        img = img.resize((self.imagesize, self.imagesize), Image.BILINEAR)
        bg = self.background
        seg, mask = segment_screw(img, bg)
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
    parser.add_argument("images", nargs="*", help="Training images")
    parser.add_argument(
        "--background",
        action="append",
        help="Background image of the empty rig; specify multiple times to build a median model",
    )
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

    dataset = ImageDataset(args.images, background=args.background)

    backbone = backbones.load("wideresnet50")
    # Dynamically determine ResNet-style layer names present in the backbone.
    layer_names = [name for name, _ in backbone.named_children() if name.startswith("layer")]
    if len(layer_names) < 3:
        raise ValueError(
            f"Backbone {backbone.__class__.__name__} lacks expected ResNet layers: {layer_names}"
        )
    layers_to_extract_from = layer_names[1:3]

    # Compute the total channel dimension produced by the selected layers so we
    # can size the embedding layers accordingly.
    with torch.no_grad():
        feats = []
        hooks = []
        module_dict = dict(backbone.named_modules())
        dummy = torch.zeros(1, 3, dataset.imagesize, dataset.imagesize)
        for name in layers_to_extract_from:
            layer = module_dict[name]
            hooks.append(layer.register_forward_hook(lambda _m, _inp, out, store=feats: store.append(out)))
        backbone(dummy)
        for h in hooks:
            h.remove()
        embed_dim = sum(f.shape[1] for f in feats)

    model = glass_mod.GLASS(device)
    # Hyperparameters align with the optimal configuration suggested in the
    # GLASS paper for unsupervised anomaly detection.
    model.load(
        backbone=backbone,
        layers_to_extract_from=layers_to_extract_from,
        device=device,
        input_shape=(3, dataset.imagesize, dataset.imagesize),
        pretrain_embed_dimension=embed_dim,
        target_embed_dimension=embed_dim,
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
    model.trainer(dataloaders["training"], dataloaders["testing"], "custom")

    ckpt = Path(model.ckpt_dir) / "ckpt.pth"
    if args.output.startswith("gs://"):
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket_name, blob_name = args.output[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_filename(str(ckpt))
    else:
        shutil.copy(ckpt, args.output)
    if dataset.background is not None:
        bg_out = f"{args.output}.background.png"
        if bg_out.startswith("gs://"):
            from google.cloud import storage  # type: ignore

            client = storage.Client()
            bucket_name, blob_name = bg_out[5:].split("/", 1)
            bucket = client.bucket(bucket_name)
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                dataset.background.save(tmp.name)
                bucket.blob(blob_name).upload_from_filename(tmp.name)
        else:
            dataset.background.save(bg_out)

    print(json.dumps({"modelId": str(args.output)}))


if __name__ == "__main__":
    main()
