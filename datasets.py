"""Dataset + transforms for domain-generalization DR grading.

The key design choice for this project: every sample carries a `domain` label
(which dataset it came from) alongside its DR `grade`. That single column is
what lets you do leave-one-domain-out training and measure the cross-domain
gap — the whole point of the research contribution.

Expected unified CSV (produced by scripts/prepare_data.py), one row per image:

    image_path,grade,domain,split
    processed/aptos/abc123.png,2,aptos,train
    processed/eyepacs/def456.png,0,eyepacs,train
    processed/messidor2/ghi789.png,3,messidor2,test

`grade` is the 0-4 DR severity scale shared across these datasets.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ImageNet stats — backbones from `timm` are pretrained with these.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def train_transforms(size: int = 512) -> A.Compose:
    """Augmentation for training.

    The colour/geometry jitter here is doing double duty: it's standard
    regularization AND a cheap form of domain randomization. When you get to
    method step 2 (heavy augmentation), this is the function you crank up.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def eval_transforms(size: int = 512) -> A.Compose:
    """Deterministic transform for validation / cross-domain test."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


class DRDataset(Dataset):
    """Diabetic-retinopathy grading dataset with domain labels.

    Args:
        df: dataframe with at least `image_path`, `grade`, `domain` columns.
        transforms: an albumentations Compose (use train/eval_transforms).
        domain_to_idx: optional mapping domain-name -> int, so domain ids are
            consistent across train/test splits. Built automatically if None.
    """

    def __init__(self, df: pd.DataFrame, transforms: A.Compose,
                 domain_to_idx: dict[str, int] | None = None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        if domain_to_idx is None:
            domain_to_idx = {d: i for i, d in enumerate(sorted(df["domain"].unique()))}
        self.domain_to_idx = domain_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing image: {row['image_path']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        grade = torch.tensor(int(row["grade"]), dtype=torch.long)
        domain = torch.tensor(self.domain_to_idx[row["domain"]], dtype=torch.long)
        return image, grade, domain


def make_dg_splits(df: pd.DataFrame, test_domains: list[str]):
    """Leave-domain(s)-out split for domain generalization.

    Everything NOT in `test_domains` becomes the source (train) pool; the held
    out domain(s) become the unseen test set the model never sees in training.
    Returns (train_df, test_df).
    """
    is_test = df["domain"].isin(test_domains)
    train_df = df[~is_test].copy()
    test_df = df[is_test].copy()
    if len(test_df) == 0:
        raise ValueError(f"No rows for test_domains={test_domains}. Available: "
                         f"{sorted(df['domain'].unique())}")
    return train_df, test_df
