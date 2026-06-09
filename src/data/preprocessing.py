"""Fundus image preprocessing.

Retinal photos arrive with big black borders, wildly different lighting, and
different resolutions per camera/dataset. Normalizing all of that *before*
training is one of the cheapest ways to reduce domain shift, so it matters a
lot for this project specifically.

Pipeline per image:
    load (RGB) -> crop dark border -> (optional) circle crop -> resize ->
    (optional) Ben-Graham color normalization

Usage:
    from src.data.preprocessing import preprocess_image
    img = preprocess_image("path/to/fundus.jpg", size=512)   # returns HxWx3 uint8 RGB
"""

from __future__ import annotations

import cv2
import numpy as np


def load_rgb(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 array (OpenCV loads BGR by default)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_dark_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop away the near-black border around the circular retina region.

    `tol` is the brightness threshold below which a pixel counts as background.
    If the image is almost entirely dark (a bad capture), the original is
    returned unchanged so nothing crashes downstream.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def circle_crop(img: np.ndarray) -> np.ndarray:
    """Mask everything outside the inscribed circle of the (square-ish) crop.

    Removes the four black corners so the model can't cheat on corner artifacts
    that differ between datasets.
    """
    h, w = img.shape[:2]
    radius = min(h, w) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)


def ben_graham(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Ben-Graham color normalization (the classic Kaggle-DR trick).

    Subtracts a heavily blurred copy of the image to flatten lighting/colour
    differences between cameras and pop the vessels, microaneurysms and
    hemorrhages. Returns a uint8 RGB image.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img, 4, blurred, -4, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_image(
    path: str,
    size: int = 512,
    do_circle_crop: bool = True,
    do_ben_graham: bool = True,
) -> np.ndarray:
    """Full per-image preprocessing. Returns an (size, size, 3) uint8 RGB array.

    Keep `do_ben_graham` toggleable: it's a strong baseline but you may want to
    ablate it as part of your domain-generalization experiments.
    """
    img = load_rgb(path)
    img = crop_dark_border(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if do_circle_crop:
        img = circle_crop(img)
    if do_ben_graham:
        img = ben_graham(img)
    return img


if __name__ == "__main__":
    # Quick self-test on a synthetic "fundus": a bright disc on a black field.
    canvas = np.zeros((900, 1200, 3), dtype=np.uint8)
    cv2.circle(canvas, (600, 450), 400, (180, 90, 60), thickness=-1)
    # Save synthetic and round-trip through preprocess_image to confirm it runs.
    cv2.imwrite("/tmp/_synthetic_fundus.jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    result = preprocess_image("/tmp/_synthetic_fundus.jpg", size=512)
    print("OK — output shape:", result.shape, "dtype:", result.dtype)
