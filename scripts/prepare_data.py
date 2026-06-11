"""Prepare raw DR datasets into a single unified, preprocessed corpus.

Turns several differently-organized datasets into ONE folder of cleaned images
plus ONE CSV with a consistent schema (image_path, grade, domain, split). That
unified CSV is what every downstream training/eval script reads.

Run from the repo root, pointing --raw-root at /kaggle/input so BOTH the
competition data and the dataset mirrors are reachable:

    python scripts/prepare_data.py --raw-root /kaggle/input \
        --out-root /kaggle/working/processed --size 512

Sources -> domain label:
    APTOS 2019 (competition)              -> domain="aptos"   (source / training)
    IDRiD (mariaherrerot/idrid-dataset)   -> domain="idrid"   (held-out target)
    EyePACS (competition, optional)       -> domain="eyepacs" (source / training)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import preprocess_image  # noqa: E402


def _save(processed_img, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))


def prepare_aptos(raw_root: Path, out_root: Path, size: int) -> list[dict]:
    """APTOS 2019 competition. train.csv: [id_code, diagnosis(0-4)]; images .png."""
    base = raw_root / "competitions" / "aptos2019-blindness-detection"
    csv_path = base / "train.csv"
    img_dir = base / "train_images"
    if not csv_path.exists():
        print(f"[skip] APTOS not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="aptos"):
        if pd.isna(r.get("id_code")) or pd.isna(r.get("diagnosis")):
            continue
        src = img_dir / f"{r['id_code']}.png"
        if not src.exists():
            continue
        out_path = out_root / "aptos" / f"{r['id_code']}.png"
        _save(preprocess_image(str(src), size=size), out_path)
        rows.append({"image_path": str(out_path), "grade": int(r["diagnosis"]),
                     "domain": "aptos", "split": "train"})
    return rows


def prepare_idrid(raw_root: Path, out_root: Path, size: int) -> list[dict]:
    """IDRiD held-out target. idrid_labels.csv: [id_code, diagnosis(0-4), ...junk];
    images are .jpg under Imagenes/Imagenes/."""
    base = raw_root / "datasets" / "mariaherrerot" / "idrid-dataset"
    csv_path = base / "idrid_labels.csv"
    img_dir = base / "Imagenes" / "Imagenes"
    if not csv_path.exists():
        print(f"[skip] IDRiD not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="idrid"):
        if pd.isna(r.get("id_code")) or pd.isna(r.get("diagnosis")):
            continue
        src = img_dir / f"{r['id_code']}.jpg"
        if not src.exists():
            continue
        out_path = out_root / "idrid" / f"{r['id_code']}.png"
        _save(preprocess_image(str(src), size=size), out_path)
        rows.append({"image_path": str(out_path), "grade": int(r["diagnosis"]),
                     "domain": "idrid", "split": "test"})
    return rows


def prepare_eyepacs(raw_root: Path, out_root: Path, size: int) -> list[dict]:
    """EyePACS (optional). trainLabels.csv: [image, level(0-4)]; images .jpeg.
    Adjust `base` to match whatever you attach (competition vs resized mirror)."""
    base = raw_root / "competitions" / "diabetic-retinopathy-detection"
    csv_path = base / "trainLabels.csv"
    img_dir = base / "train"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/kaggle/input")
    ap.add_argument("--out-root", default="/kaggle/working/processed")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    raw_root, out_root = Path(args.raw_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows += prepare_aptos(raw_root, out_root, args.size)
    rows += prepare_idrid(raw_root, out_root, args.size)
    rows += prepare_eyepacs(raw_root, out_root, args.size)

    if not rows:
        print("No data processed. Are APTOS / IDRiD attached, and is "
              "--raw-root pointing at /kaggle/input ?")
        return

    df = pd.DataFrame(rows)
    out_csv = out_root / "labels.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")
    print(df.groupby(["domain", "grade"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()