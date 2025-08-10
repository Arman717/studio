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


def segment_with_sam(img: Image.Image) -> Image.Image:
    """Use Segment Anything (v2 if installed) to remove the background."""
    global _sam_generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _sam_generator is None:
        if SAM2_AVAILABLE:
            checkpoint = Path(__file__).resolve().parent / "sam2.1_hiera_large.pt"
            cfg = Path(__file__).resolve().parent / "sam2.1_hiera_l.yaml"
            if not checkpoint.exists() or not cfg.exists():
                import urllib.request

                if not checkpoint.exists():
                    url_ckpt = (
                        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
                    )
                    urllib.request.urlretrieve(url_ckpt, checkpoint)
                if not cfg.exists():
                    url_cfg = (
                        "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
                    )
                    urllib.request.urlretrieve(url_cfg, cfg)
            sam = build_sam2(str(cfg), str(checkpoint), device=device, apply_postprocessing=False)
            _sam_generator = SAM2AutomaticMaskGenerator(sam)
        else:
            checkpoint = Path(__file__).resolve().parent / "sam_vit_h_4b8939.pth"
            if not checkpoint.exists():
                import urllib.request

                url = (
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                )
                urllib.request.urlretrieve(url, checkpoint)
            sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
            sam.to(device)
            _sam_generator = SamAutomaticMaskGenerator(sam)
    np_img = np.array(img)
    masks = _sam_generator.generate(np_img)
    if not masks:
        mask = np.ones(np_img.shape[:2], dtype=bool)
    else:
        mask = max(masks, key=lambda m: m.get("area", 0))["segmentation"]
    rgb = np_img.copy()
    rgb[~mask] = 0
    return Image.fromarray(rgb)


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.imagesize = 288
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
        img = Image.open(self.path).convert("RGB")
        img = segment_with_sam(img)
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
    img = segment_with_sam(Image.open(args.image).convert("RGB"))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    cmap = plt.get_cmap("viridis")
    colored = (cmap(mask)[:, :, :3] * 255).astype("uint8")
    heat_img = Image.fromarray(colored).resize(img.size)
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


if __name__ == "__main__":
    main()
