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


def remove_background(img: Image.Image):
    gray = img.convert("L")
    arr = np.array(gray)
    hist = np.bincount(arr.flatten(), minlength=256)
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    var_max = 0.0
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
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = i
    mask = arr < threshold
    rgb = np.array(img)
    rgb[~mask] = 0
    return Image.fromarray(rgb)


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.imagesize = 288
        self.distribution = 0
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
        img = Image.open(self.path).convert("RGB")
        img = remove_background(img)
        img = img.resize((self.imagesize, self.imagesize))
        return {
            "image": self.tf(img),
            "is_anomaly": torch.tensor(0),
            "mask_gt": torch.zeros(1, self.imagesize, self.imagesize),
            "image_path": str(self.path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze screw image with GLASS")
    parser.add_argument("--image", required=True, help="Path to screw image")
    parser.add_argument("--model", required=True, help="Path to trained model")
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

    img = remove_background(Image.open(args.image).convert("RGB"))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    cmap = plt.get_cmap("viridis")
    colored = (cmap(mask)[:, :, :3] * 255).astype("uint8")
    heat_img = Image.fromarray(colored).resize(img.size)
    overlay = Image.blend(img, heat_img, alpha=0.5)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = {
        "defectDetected": score > 0.5,
        "defectVisualizationDataUri": f"data:image/png;base64,{b64}",
        "screwStatus": "NOK" if score > 0.5 else "OK",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
