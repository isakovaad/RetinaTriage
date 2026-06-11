# RetinaTriage

**Participant:** Dilbar Isakova
**Program:** AI Mentorship Program — 4-week capstone (Option B, own project)

A clinician-facing decision-support tool for diabetic retinopathy (DR). Upload a
retinal (fundus) photo and get back a DR severity grade, a Grad-CAM heatmap showing
where the model looked, a calibrated confidence, and a referral flag. Built with a
research focus on **domain generalization** — staying accurate on images from cameras
and clinics the model never trained on.

> **Not a medical device.** Research/educational prototype only — see disclaimer below.

## What it is

Two halves, kept separate on purpose:

- **Product** — a (planned) responsive web app: upload fundus photo → grade + heatmap
  + confidence + referral. A triage / decision-support aid, **not** autonomous diagnosis.
- **Research** — a reproducible domain-generalization benchmark: train on source
  datasets, evaluate on an *unseen* target, measure and reduce the cross-domain gap.

For: primary-care / screening settings without an on-site ophthalmologist, and the
medical-imaging ML community.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Model | PyTorch · timm (ResNet50) · albumentations |
| Training | Kaggle (T4 GPU) · scikit-learn (metrics) · Weights & Biases |
| Serving | FastAPI · Uvicorn (CPU inference) |
| Frontend *(planned)* | Next.js / React (responsive) |
| Auth & Payments *(planned)* | Supabase or Clerk · Stripe (test mode) |
| Deploy *(planned)* | Render (API) · Vercel (frontend) · GitHub Actions CI |

## Current status

- [x] Data pipeline — APTOS 2019 preprocessed into a unified schema with a `domain` label
- [x] Baseline model — fine-tuned ResNet50, in-domain validation quadratic weighted kappa ≈ **0.87**
- [x] Serving — FastAPI `/predict` returns grade + confidence + referral
- [ ] Cross-domain evaluation — held-out dataset → first generalization-gap number *(in progress)*
- [ ] Frontend (Next.js) + auth · payments · monitoring · CI/CD · 50-sample eval

## Prerequisites

- Python 3.11+
- A (free) Kaggle account — for the APTOS dataset and a T4 GPU (data prep + training)
- The trained weights file `baseline_resnet50.pt` — required to run the backend
- *(Planned)* Node.js 20+ for the frontend

## Getting Started

### 1. Prepare the data (on Kaggle)
Attach the **APTOS 2019** competition (Add Input → Competitions tab), clone this repo
into the notebook, then:
```bash
python scripts/prepare_data.py \
  --raw-root /kaggle/input/competitions \
  --out-root /kaggle/working/processed --size 512
```

### 2. Train the baseline (Kaggle, T4 GPU)
```bash
python scripts/train.py \
  --labels /kaggle/working/processed/labels.csv \
  --out baseline_resnet50.pt --epochs 5 --img-size 384 --batch-size 16
```
Add `--test-domains <name>` once a second dataset is folded in, to print the
cross-domain kappa and the generalization gap.

### 3. Serve the model (locally)
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place baseline_resnet50.pt in backend/
uvicorn main:app --reload --port 8000
```
Open <http://localhost:8000/docs> and try `/predict` with a fundus image.

## Project Structure

```
RetinaTriage/
├── src/
│   └── data/
│       ├── preprocessing.py   # fundus crop, resize, circle-crop, color normalization
│       └── datasets.py        # PyTorch dataset + domain-aware leave-one-domain-out split
├── scripts/
│   ├── prepare_data.py        # raw datasets → one unified labels.csv
│   └── train.py               # fine-tune baseline, eval (in-domain + cross-domain)
├── backend/
│   ├── main.py                # FastAPI app: /predict, /health
│   ├── inference.py           # loads .pt, reuses training preprocessing
│   └── requirements.txt
├── diary/                     # build diary (program requirement)
└── README.md
```

## Key Architectural Decisions

- **Train/serve preprocessing parity** — the backend imports the *exact* preprocessing
  and eval transform used in training, so a served image is processed identically to a
  trained one. This keeps live accuracy matching validation accuracy.
- **Domain-aware dataset** — every image carries a `domain` label, and `make_dg_splits`
  does leave-one-domain-out. This single design choice is what makes the cross-domain
  benchmark possible.
- **Quadratic weighted kappa** as the metric — the right choice for *ordinal* DR grades;
  it penalizes calling a grade-4 a grade-0 far more than a grade-3.
- **Class-weighted loss** — counters the heavy grade imbalance (grade 0 dominates APTOS).
- **Thin serving shell** — training happens on Kaggle and produces a portable `.pt`; the
  FastAPI backend just loads it. Single-image inference runs on CPU, so production needs no GPU.
- **Referral threshold = grade ≥ 2** — "referable DR" is moderate or worse.
- **Ben-Graham normalization is a toggle** — so it can be ablated as part of the
  domain-generalization experiments rather than baked in.

## Research: domain generalization

Train on source domain(s) (APTOS, later + EyePACS), evaluate on an *unseen* target
(Messidor-2 / IDRiD) with no target-domain data, and report in-domain vs. cross-domain
quadratic weighted kappa — the **generalization gap**. Method ladder: ERM baseline →
heavy color/Fourier augmentation → a dedicated DG method, each reported honestly.
Protocol follows the published DG-for-DR benchmark line of work so numbers are comparable.

## Commands

| Command | Description |
| --- | --- |
| `python scripts/prepare_data.py ...` | Preprocess raw datasets into a unified `labels.csv` |
| `python scripts/train.py ...` | Fine-tune the baseline, save weights, optionally run cross-domain eval |
| `uvicorn main:app --reload --port 8000` | Run the inference API locally (`/docs` for the UI) |

## Milestone plan

| Week | Milestone | Target |
| --- | --- | --- |
| 1 | Foundation | auth, data pipeline, smallest end-to-end slice (upload → grade) |
| 2 | Core Features | cross-domain eval + gap number; Grad-CAM + calibrated confidence; full upload → grade + heatmap + referral |
| 3 | Polish & Integrate | Stripe test-mode payments (see note); complete mobile-friendly UI |
| 4 | Ship It | deploy (API + frontend); CI/CD on push; monitoring; 50-sample cross-domain evaluation |

**Payment substitution note:** a screening tool's "payment" is artificial, so Week 3's
payment milestone is a Stripe **test-mode** per-screening credit / pro-tier gate on the
upload endpoint — same integration, no pretend clinical billing.

## Disclaimer

RetinaTriage is a research and educational prototype. It is **not** a medical device, is
**not** FDA/CE cleared, and must **not** be used to make clinical decisions. All data is
from publicly available, de-identified research datasets. Outputs assist, never replace,
a qualified clinician.
