#!/usr/bin/env python3
"""Run defect analysis on a single image using the GLASS model."""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

try:
    import torch
    from torchvision import transforms
except ModuleNotFoundError:
    print(
        "Missing dependencies. Please install PyTorch and torchvision as listed in GLASS requirements.",
        file=sys.stderr,
    )
    raise

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




class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.path = path
        img = Image.open(path).convert("RGB")
        seg, mask = segment_screw(img)
        if mask.max() == 0:
            mask[0, 0] = 1
        ys, xs = np.where(mask)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        pad = 10
        xmin = max(0, xmin - pad)
        ymin = max(0, ymin - pad)
        xmax = min(seg.width - 1, xmax + pad)
        ymax = min(seg.height - 1, ymax + pad)
        seg = seg.crop((xmin, ymin, xmax + 1, ymax + 1))
        size = max(seg.width, seg.height)
        canvas = Image.new("RGB", (size, size))
        left = (size - seg.width) // 2
        top = (size - seg.height) // 2
        canvas.paste(seg, (left, top))
        self.img = canvas
        self.imagesize = size
        # Match the manifold distribution used during training.
        self.distribution = 2
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return {
            "image": self.tf(self.img),
            "is_anomaly": torch.tensor(0),
            "mask_gt": torch.zeros(1, self.imagesize, self.imagesize),
            "image_path": str(self.path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze screw image with GLASS")
    parser.add_argument("--image", required=True, help="Path to screw image")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument(
        "--backbone",
        default="efficientnet_b5",
        help="GLASS backbone name (e.g. efficientnet_b5, wideresnet50)",
    )
    parser.add_argument(
        "--output", help="Optional path to save the visualization overlay"
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent / "glass_repo"
    ensure_repo(repo_dir)
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    import glass as glass_mod  # type: ignore
    import backbones  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SingleImageDataset(Path(args.image))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    backbone = backbones.load(args.backbone)
    model = glass_mod.GLASS(device)
    # Use the hyperparameters recommended by the GLASS paper so the model
    # architecture matches the training configuration.
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

    state = torch.load(args.model, map_location=device)
    if "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
        if "pre_projection" in state:
            model.pre_projection.load_state_dict(state["pre_projection"])
    else:
        model.load_state_dict(state, strict=False)

    images, scores, masks, _, _ = model.predict(loader)
    score = float(scores[0])
    mask = masks[0]
    img = dataset.img
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    cmap = plt.get_cmap("viridis")
    colored = (cmap(mask)[:, :, :3] * 255).astype("uint8")
    heat_img = Image.fromarray(colored)
    overlay = Image.blend(img, heat_img, alpha=0.5)
    if args.output:
        overlay.save(args.output)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = {
        "defectDetected": score > 0.5,
        "defectVisualizationDataUri": f"data:image/png;base64,{b64}",
        "screwStatus": "NOK" if score > 0.5 else "OK",
    }
    print(json.dumps(result))


def segment_screw(img: Image.Image):
    """Segment the screw using Otsu thresholding and return image and mask."""
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
    rgb = np_img.copy()
    rgb[~mask] = 0
    return Image.fromarray(rgb), mask.astype(np.uint8)

if __name__ == "__main__":
    main()

