"""Train a baseline DR grading model and evaluate it.

Run from the repo root (the folder containing src/ and scripts/):

    python scripts/train.py \
        --labels /kaggle/working/processed/labels.csv \
        --out /kaggle/working/baseline_resnet50.pt \
        --epochs 5 --img-size 384 --batch-size 16

Right now you have one domain (aptos), so this trains on aptos and reports an
IN-DOMAIN validation quadratic weighted kappa, then saves the best weights.

Once you add a second dataset as a held-out domain (e.g. messidor2 or idrid),
re-run with `--test-domains messidor2` to also get the CROSS-DOMAIN kappa and
the generalization gap — no other change needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datasets import (  # noqa: E402
    DRDataset, train_transforms, eval_transforms, make_dg_splits,
)

NUM_CLASSES = 5


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return quadratic weighted kappa over a loader."""
    model.eval()
    preds, targets = [], []
    for images, grades, _ in loader:
        logits = model(images.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(grades.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return cohen_kappa_score(targets, preds, weights="quadratic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/kaggle/working/processed/labels.csv")
    ap.add_argument("--out", default="/kaggle/working/baseline.pt")
    ap.add_argument("--model", default="resnet50", help="any timm model name")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--test-domains", nargs="*", default=[],
                    help="domains to hold out for cross-domain eval, e.g. messidor2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df = pd.read_csv(args.labels)
    # Stable domain ids across every split.
    domain_to_idx = {d: i for i, d in enumerate(sorted(df["domain"].unique()))}

    # Hold out the test domains (if any); the rest is the training pool.
    if args.test_domains:
        train_pool, test_df = make_dg_splits(df, args.test_domains)
        print(f"held-out test domain(s): {args.test_domains} ({len(test_df)} imgs)")
    else:
        train_pool, test_df = df, None
        print("no held-out domain yet -> reporting in-domain validation only")

    # Stratified in-domain validation split for model selection.
    train_df, val_df = train_test_split(
        train_pool, test_size=0.15, stratify=train_pool["grade"], random_state=42,
    )
    print(f"train: {len(train_df)}  val: {len(val_df)}")

    def loader(frame, train: bool):
        tfm = train_transforms(args.img_size) if train else eval_transforms(args.img_size)
        ds = DRDataset(frame, tfm, domain_to_idx=domain_to_idx)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                          num_workers=2, pin_memory=(device == "cuda"))

    train_loader = loader(train_df, True)
    val_loader = loader(val_df, False)

    # Class weights to counter the imbalance (grade 0 dominates).
    counts = train_df["grade"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    counts = counts.replace(0, 1)
    weights = counts.sum() / (NUM_CLASSES * counts)
    class_weights = torch.tensor(weights.values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = timm.create_model(args.model, pretrained=True, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_kappa = -1.0
    for epoch in range(args.epochs):
        model.train()
        for images, grades, _ in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            images, grades = images.to(device), grades.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(images), grades)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_kappa = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}: val quad-kappa = {val_kappa:.4f}")
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save({"state_dict": model.state_dict(),
                        "model_name": args.model,
                        "num_classes": NUM_CLASSES,
                        "img_size": args.img_size}, args.out)
            print(f"  saved new best -> {args.out}")

    print(f"\nbest in-domain validation kappa: {best_kappa:.4f}")

    # Cross-domain evaluation, if a held-out domain was provided.
    if test_df is not None:
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_kappa = evaluate(model, loader(test_df, False), device)
        print(f"cross-domain kappa on {args.test_domains}: {test_kappa:.4f}")
        print(f"generalization gap: {best_kappa - test_kappa:.4f}")


if __name__ == "__main__":
    main()
