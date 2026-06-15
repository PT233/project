# Architecture

System structure for the cancer histology classification project. This document
describes the file layout and the responsibility of each module.

## File and directory layout

```
project/
├── data/
│   ├── breakhis/                  # BreaKHis images, 40x (not tracked in git)
│   ├── dlbcl/                     # DLBCL patches + clinical CSV (not tracked in git)
│   └── splits/                    # split CSV files (tracked in git)
│       ├── breakhis_train.csv
│       ├── breakhis_val.csv
│       ├── breakhis_test.csv
│       ├── dlbcl_survival_train.csv
│       ├── dlbcl_survival_val.csv
│       └── dlbcl_survival_test.csv
│
├── src/
│   ├── datasets/
│   │   ├── base_dataset.py        # abstract dataset contract
│   │   ├── breakhis_dataset.py    # BreaKHis 40x patch dataset
│   │   ├── dlbcl_dataset.py       # DLBCL patch dataset
│   │   └── dlbcl_bag_dataset.py   # DLBCL patient-bag dataset for MIL
│   ├── models/
│   │   ├── classifier.py          # EfficientNet-B2 classifier factory
│   │   ├── gradcam.py             # Grad-CAM overlay generation
│   │   └── mil.py                 # EfficientNet-B2 + gated attention MIL (ABMIL)
│   ├── training/
│   │   ├── train.py               # patch-level training loop (AMP)
│   │   ├── train_mil.py           # bag-level ABMIL training loop
│   │   └── evaluate.py            # checkpoint evaluation, writes JSON
│   └── utils/
│       ├── augmentations.py       # albumentations transform pipelines
│       ├── logger.py              # Weights & Biases wrapper
│       └── patient_aggregation.py # patch-to-patient probability aggregation
│
├── configs/
│   ├── breakhis.yaml
│   ├── dlbcl.yaml
│   └── dlbcl_mil.yaml
│
├── scripts/
│   └── build_survival_split.py    # builds the DLBCL survival split (seed=42)
│
├── artifacts/                     # checkpoints, eval JSON, wandb runs (not tracked in git)
├── requirements.txt
├── README.md
├── architecture.md
├── DLBCL_MIL_RESULT.md
└── .gitignore
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `datasets/base_dataset.py` | Abstract base class fixing the sample dict returned by every patch dataset (`image`, `label`, `patient_id`, `meta`). |
| `datasets/breakhis_dataset.py` | Reads the BreaKHis CSV, keeps 40x rows only, returns standardised image samples. |
| `datasets/dlbcl_dataset.py` | Reads the DLBCL CSV, filters out rows whose image file is missing, returns patch samples, and exposes patient-to-index lookups. |
| `datasets/dlbcl_bag_dataset.py` | Wraps the DLBCL patch dataset and emits one bag of sampled patches per patient for MIL (random sampling in train, deterministic in eval). |
| `models/classifier.py` | Builds an EfficientNet-B2 (via timm) with a fresh linear head; loads ImageNet weights from the torch hub cache. |
| `models/gradcam.py` | Runs Grad-CAM on a single image using the last EfficientNet-B2 block and writes an overlay PNG. |
| `models/mil.py` | EfficientNet-B2 backbone plus a gated-attention pooling head; produces a bag-level logit and the per-instance attention weights. |
| `training/train.py` | Patch-level training loop for BreaKHis and the DLBCL baseline; AMP, cosine schedule, checkpointing, optional patient-level validation AUC. |
| `training/train_mil.py` | Bag-level ABMIL training loop; selects on validation patient AUC, evaluates the best checkpoint on the test split, and appends results to `DLBCL_MIL_RESULT.md`. |
| `training/evaluate.py` | Loads a checkpoint, runs inference on a split, computes image-level or patient-level metrics, and writes a JSON result. |
| `utils/augmentations.py` | Train/val transform pipelines (albumentations); separate profiles for BreaKHis and the stronger H&E-safe DLBCL augmentation. |
| `utils/logger.py` | Thin wrapper around Weights & Biases init/log/finish, run directory under `artifacts/wandb/`. |
| `utils/patient_aggregation.py` | Aggregates patch probabilities into a patient score (max / mean / top-k mean / percentile) and computes patient-level metrics. |
| `configs/*.yaml` | Hyper-parameters and data paths; edited instead of code to change an experiment. |
| `scripts/build_survival_split.py` | Builds the DLBCL survival split from the clinical CSV with patient-level separation (seed=42). |

## Configurations

| Config | Purpose |
|---|---|
| `configs/breakhis.yaml` | BreaKHis 40x patch-level training. |
| `configs/dlbcl.yaml` | DLBCL patch-level baseline; selects on patient-level AUC via max aggregation. |
| `configs/dlbcl_mil.yaml` | DLBCL end-to-end ABMIL; bag size, attention hidden width, and dropout. |

## Data splits

Split CSV files live in `data/splits/` and are tracked in git; the image data is
not. The DLBCL survival split holds 71 train / 15 validation / 16 test patients
(11,334 / 2,668 / 3,072 patches), produced by `scripts/build_survival_split.py`
with patient-level separation so no patient appears in more than one split.

## State and outputs

| Item | Location |
|---|---|
| Raw images | local filesystem under `data/` (not tracked in git) |
| Split CSVs | `data/splits/*.csv` (tracked in git) |
| Model checkpoints | `artifacts/results/checkpoints/*.pth` (not tracked in git) |
| Evaluation JSON | `artifacts/results/*_eval.json` |
| ABMIL test results | `DLBCL_MIL_RESULT.md` (appended by `train_mil.py`) |
| Training logs / metrics | Weights & Biases (offline runs under `artifacts/wandb/`) |
| Pretrained weights | torch hub cache (`~/.cache/torch/hub/checkpoints/`) |

## Data flow

```
                 build_survival_split.py
                          │
                          ▼
   data/splits/*.csv ──► datasets ──► models ──► training ──► checkpoints
                          │              │           │
                   augmentations    classifier    train.py ──► evaluate.py ──► *_eval.json
                                     mil (ABMIL)   train_mil.py ──► DLBCL_MIL_RESULT.md
                                     gradcam
```

Patch-level training (`train.py`) consumes `breakhis_dataset` or `dlbcl_dataset`
and the `classifier` model; for DLBCL it reports a patient-level AUC through
`patient_aggregation`. Bag-level training (`train_mil.py`) consumes
`dlbcl_bag_dataset` and the `mil` model and reports metrics directly at the
patient level.

## Design notes

| Decision | Choice | Reason |
|---|---|---|
| Backbone | EfficientNet-B2 (timm) | Good accuracy-to-memory trade-off for an 8 GB GPU; larger variants risk OOM. |
| BreaKHis magnification | 40x only | Single magnification keeps data volume and epoch time manageable. |
| Mixed precision | AMP enabled | Larger effective batch size without loss of convergence. |
| DLBCL patient score | Patch-probability aggregation (max default) | Simple, no extra parameters; max is sensitive to focal positive patches. |
| DLBCL end-to-end model | Gated-attention MIL | Learns which patches matter for the patient label instead of fixing the aggregation rule. |
| Configuration | YAML files | Hyper-parameters change without touching code. |
| Experiment tracking | Weights & Biases (offline) | Local run capture without requiring network access. |
