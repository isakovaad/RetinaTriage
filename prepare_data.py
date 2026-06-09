"""Prepare raw DR datasets into a single unified, preprocessed corpus.

This turns several differently-organized datasets into ONE folder of cleaned
images plus ONE CSV with a consistent schema (image_path, grade, domain, split).
That unified CSV is what every downstream training/eval script reads.

------------------------------------------------------------------------------
STEP 0 — get the raw data (do this once, manually)
------------------------------------------------------------------------------
On Kaggle/Colab, set up the Kaggle API (Account -> Create New API Token gives
kaggle.json), then:

    pip install kaggle
    mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

    # APTOS 2019 (~3.6k images, train.csv has the 0-4 grades)
    kaggle competitions download -c aptos2019-blindness-detection -p data/raw/aptos

    # EyePACS (the big one; "Diabetic Retinopathy Detection")
    kaggle competitions download -c diabetic-retinopathy-detection -p data/raw/eyepacs

Messidor-2 / IDRiD are downloaded from their own project pages (registration
required) — drop them under data/raw/messidor2 and data/raw/idrid. Keep these
as your HELD-OUT test domains.

NOTE: this script does not auto-download (Kaggle needs your credentials). It
expects the raw folders to already exist, then preprocesses them.

------------------------------------------------------------------------------
STEP 1 — run this script
------------------------------------------------------------------------------
    python scripts/prepare_data.py --raw-root data/raw --out-root processed --size 512

Output:
    processed/<domain>/<image_id>.png      cleaned images
    processed/labels.csv                   image_path,grade,domain,split
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

# Make `src` importable when running from repo root.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import preprocess_image  # noqa: E402


def _save(processed_img, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))


def prepare_aptos(raw_root: Path, out_root: Path, size: int) -> list[dict]:
    """APTOS: train.csv has columns [id_code, diagnosis(0-4)]; images in train_images/."""
    csv_path = raw_root / "aptos" / "train.csv"
    img_dir = raw_root / "aptos" / "train_images"
    if not csv_path.exists():
        print(f"[skip] APTOS not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="aptos"):
        src = img_dir / f"{r['id_code']}.png"
        if not src.exists():
            continue
        out_path = out_root / "aptos" / f"{r['id_code']}.png"
        _save(preprocess_image(str(src), size=size), out_path)
        rows.append({"image_path": str(out_path), "grade": int(r["diagnosis"]),
                     "domain": "aptos", "split": "train"})
    return rows


def prepare_eyepacs(raw_root: Path, out_root: Path, size: int) -> list[dict]:
    """EyePACS: trainLabels.csv has [image, level(0-4)]; images are .jpeg."""
    csv_path = raw_root / "eyepacs" / "trainLabels.csv"
    img_dir = raw_root / "eyepacs" / "train"
    if not csv_path.exists():
        print(f"[skip] EyePACS not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="eyepacs"):
        src = img_dir / f"{r['image']}.jpeg"
        if not src.exists():
            continue
        out_path = out_root / "eyepacs" / f"{r['image']}.png"
        _save(preprocess_image(str(src), size=size), out_path)
        rows.append({"image_path": str(out_path), "grade": int(r["level"]),
                     "domain": "eyepacs", "split": "train"})
    return rows


# Messidor-2 / IDRiD loaders are left as a deliberate TODO: their label files
# differ in format, and wiring one of them up yourself is a good Week-1 exercise
# (and good diary material). They become domain="messidor2"/"idrid", split="test".


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out-root", default="processed")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    raw_root, out_root = Path(args.raw_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows += prepare_aptos(raw_root, out_root, args.size)
    rows += prepare_eyepacs(raw_root, out_root, args.size)

    if not rows:
        print("No data processed. Did you download the raw datasets first? "
              "See the instructions at the top of this file.")
        return

    df = pd.DataFrame(rows)
    out_csv = out_root / "labels.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")
    print(df.groupby(["domain", "grade"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
