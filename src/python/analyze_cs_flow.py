#!/usr/bin/env python3
"""Run defect analysis on a single image using CS-Flow."""

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
except ModuleNotFoundError as e:
    print(
        "Missing dependencies. Please install PyTorch and torchvision as listed in cs-flow requirements.",
        file=sys.stderr,
    )
    raise


def ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    subprocess.check_call(
        ["git", "clone", "https://github.com/Arman717/cs-flow", str(repo_dir)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze screw image with cs-flow")
    parser.add_argument("--image", required=True, help="Path to screw image")
    parser.add_argument("--model", required=True, help="Path to trained model")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent / "cs_flow_repo"
    ensure_repo(repo_dir)
    sys.path.insert(0, str(repo_dir))
    # ensure model paths resolve correctly
    os.chdir(repo_dir)

    import config as c  # type: ignore
    import freia_funcs  # type: ignore
    from model import FeatureExtractor, nf_forward  # type: ignore

    c.device = "cpu"
    c.pre_extracted = False

    from torch.serialization import add_safe_globals  # type: ignore
    from contextlib import redirect_stdout

    # allow loading the custom ReversibleGraphNet class used by CS-Flow
    add_safe_globals([freia_funcs.ReversibleGraphNet])

    model_file = Path("models") / "tmp" / Path(args.model).name
    with redirect_stdout(sys.stderr):
        model = torch.load(model_file, weights_only=False)
    model.eval()
    model.to("cpu")

    fe = FeatureExtractor()
    fe.eval()

    tf = transforms.Compose(
        [
            transforms.Resize(c.img_size),
            transforms.ToTensor(),
            transforms.Normalize(c.norm_mean, c.norm_std),
        ]
    )

    img = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        feats = [fe.eff_ext(tf(img).unsqueeze(0))]
        z, jac = nf_forward(model, feats)
        score = torch.mean(z[0] ** 2 / 2).item()

    result = {
        "defectDetected": score > 2.5,
        "defectVisualizationDataUri": "",
        "screwStatus": "NOK" if score > 2.5 else "OK",
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
