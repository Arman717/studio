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
from typing import Optional, Tuple

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




class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, background: Optional[Image.Image] = None):
        self.path = path
        img = Image.open(path).convert("RGB")
        # Resize to a square so analysis can handle non-square inputs
        size = min(img.width, img.height)
        img = img.resize((size, size), Image.BILINEAR)
        bg = background.resize((size, size), Image.BILINEAR) if background is not None else None
        self.img, _ = segment_screw(img, bg)
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
        "--output", help="Optional path to save the visualization overlay"
    )
    parser.add_argument(
        "--background",
        action="append",
        help="Background image of the empty rig; specify multiple times to build a median model",
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

    bg_paths = args.background
    if bg_paths is None:
        default_bg = Path(f"{args.model}.background.png")
        if default_bg.exists():
            bg_paths = [str(default_bg)]
    background_img = None
    if bg_paths:
        bg_arrays = [np.array(Image.open(p).convert("RGB"), dtype=np.uint8) for p in bg_paths]
        median = np.median(np.stack(bg_arrays, axis=0), axis=0).astype(np.uint8)
        background_img = Image.fromarray(median)

    dataset = SingleImageDataset(Path(args.image), background_img)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    backbone = backbones.load("wideresnet50")
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


def segment_screw(
    img: Image.Image,
    background: Optional[Image.Image] = None,
    output_size: Optional[int] = None,
    threshold: int = 20,
):
    """Segment the screw by subtracting a background model and thresholding.

    ``background`` is a median-composited image of the empty rig. ``output_size``
    optionally resizes the full image and mask to this size.
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

if __name__ == "__main__":
    main()

