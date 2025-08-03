#!/usr/bin/env python3
"""Utility script that wraps the CS-Flow training pipeline."""

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
    from torchvision import transforms  # noqa: F401
except ModuleNotFoundError:
    print(
        "Missing dependencies. Please install PyTorch and torchvision as listed in cs-flow requirements.",
        file=sys.stderr,
    )
    raise

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    print(
        "Missing dependencies. Please install opencv-python-headless and numpy.",
        file=sys.stderr,
    )
    raise


def ensure_repo(repo_dir: Path) -> None:
    """Clone the cs-flow repository if it doesn't exist."""
    if repo_dir.exists():
        return
    subprocess.check_call(
        ["git", "clone", "https://github.com/Arman717/cs-flow", str(repo_dir)]
    )


def patch_training_code(repo_dir: Path) -> None:
    """Patch cs-flow to avoid roc_auc_score errors when only one class is present."""
    train_py = repo_dir / "train.py"
    text = train_py.read_text()
    if "safe_roc_auc_score" in text:
        return
    replacement = (
        "from sklearn.metrics import roc_auc_score\n"
        "from sklearn.exceptions import UndefinedMetricWarning\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore', category=UndefinedMetricWarning)\n\n"
        "def safe_roc_auc_score(y_true, y_score):\n"
        "    try:\n"
        "        return roc_auc_score(y_true, y_score)\n"
        "    except ValueError:\n"
        "        # dataset may contain only one class; return neutral AUROC\n"
        "        return 0.5\n"
    )
    text = text.replace(
        "from sklearn.metrics import roc_auc_score", replacement, 1
    )
    text = text.replace("roc_auc_score(is_anomaly, anomaly_score)", "safe_roc_auc_score(is_anomaly, anomaly_score)")
    train_py.write_text(text)


def segment_screw(src: Path, dest: Path) -> None:
    """Segment the screw from the background using simple thresholding."""
    img = cv2.imread(str(src))
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        cv2.imwrite(str(dest), img)
        return
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(str(dest), segmented)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cs-flow model")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("images", nargs="*", help="Training images")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent / "cs_flow_repo"
    ensure_repo(repo_dir)
    patch_training_code(repo_dir)

    sys.path.insert(0, str(repo_dir))
    # run training inside the repo directory so relative paths match
    os.chdir(repo_dir)
    import config as c  # type: ignore
    from train import train as cs_train  # type: ignore
    from utils import load_datasets, make_dataloaders  # type: ignore

    c.device = "cuda" if torch.cuda.is_available() else "cpu"
    c.dataset_path = tempfile.mkdtemp()
    c.class_name = "custom"
    c.modelname = Path(args.output).name
    c.pre_extracted = False

    data_dir = Path(c.dataset_path) / c.class_name
    train_good = data_dir / "train" / "good"
    test_good = data_dir / "test" / "good"
    train_good.mkdir(parents=True, exist_ok=True)
    test_good.mkdir(parents=True, exist_ok=True)

    if len(args.images) == 0:
        print("No training images were provided", file=sys.stderr)
        sys.exit(1)

    if len(args.images) < 2:
        train_imgs = args.images
        test_imgs = args.images
    else:
        test_count = min(2, len(args.images))
        train_imgs = args.images[:-test_count]
        test_imgs = args.images[-test_count:]

    for idx, img in enumerate(train_imgs):
        dest = train_good / f"img_{idx}.png"
        shutil.copy(img, dest)
        segment_screw(dest, dest)

    for idx, img in enumerate(test_imgs):
        dest = test_good / f"img_{idx}.png"
        shutil.copy(img, dest)
        segment_screw(dest, dest)

    train_set, test_set = load_datasets(c.dataset_path, c.class_name)
    train_loader, test_loader = make_dataloaders(train_set, test_set)
    # redirect training logs to stderr so stdout contains only JSON
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):
        cs_train(train_loader, test_loader)

    model_path = repo_dir / "models" / "tmp" / c.modelname
    shutil.copy(model_path, args.output)

    print(json.dumps({"modelId": str(args.output)}))


if __name__ == '__main__':
    main()
