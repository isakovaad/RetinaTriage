"""Load the trained DR model and run inference on an uploaded image.

This reuses the EXACT training preprocessing (src/data/preprocessing.py) and the
eval transform (src/data/datasets.py), so the model sees an image at serving
time the same way it did during training. That train/serve consistency is what
keeps the deployed accuracy matching your validation accuracy.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import timm

# Add the repo root (one level up from backend/) so `src` is importable.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import preprocess_image  # noqa: E402
from src.data.datasets import eval_transforms  # noqa: E402

GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
REFER_FROM_GRADE = 2   # "referable DR" = moderate or worse
PREP_SIZE = 512        # must match the --size used in prepare_data.py

# Drop your downloaded weights here, or set MODEL_PATH in the environment.
MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "baseline_resnet50.pt"))

_model = None
_img_size = 384


def load_model(path: str = MODEL_PATH):
    """Rebuild the timm model from the checkpoint and load the trained weights."""
    global _model, _img_size
    ckpt = torch.load(path, map_location="cpu")
    _img_size = ckpt.get("img_size", 384)
    model = timm.create_model(ckpt["model_name"], pretrained=False,
                              num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _model = model
    return model


def predict_from_bytes(image_bytes: bytes) -> dict:
    """Take raw image bytes, return the grade, confidence, and a referral flag."""
    if _model is None:
        load_model()

    # Write to a temp file so we can reuse preprocess_image unchanged.
    # delete=False + manual unlink keeps this working on Windows too.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()
        img = preprocess_image(tmp.name, size=PREP_SIZE)  # RGB uint8, exactly as in training
    finally:
        os.unlink(tmp.name)

    tensor = eval_transforms(_img_size)(image=img)["image"].unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]

    grade = int(probs.argmax())
    return {
        "grade": grade,
        "label": GRADE_LABELS[grade],
        "confidence": round(float(probs[grade]), 4),
        "refer": grade >= REFER_FROM_GRADE,
        "probabilities": {GRADE_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }
