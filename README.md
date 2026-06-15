# Cancer Histology Classification

Binary classification of histology images with EfficientNet-B2, evaluated on two
cohorts: **BreaKHis** (breast tumour, benign vs. malignant, 40x) and **DLBCL**
(diffuse large B-cell lymphoma, survival). For DLBCL we also implement an
end-to-end attention-based multiple-instance learning (ABMIL) model that learns
a patient-level prediction directly from a bag of patches.

## Overview

- **BreaKHis** — patch/image-level classification at 40x magnification. One
  EfficientNet-B2 backbone with a linear head, trained on individual images.
- **DLBCL baseline** — the same patch-level classifier applied to DLBCL patches;
  patient scores are obtained by aggregating patch probabilities (max pooling by
  default).
- **DLBCL ABMIL** — EfficientNet-B2 feature extractor with a gated-attention MIL
  head (Ilse et al., ICML 2018). Each patient is one bag of sampled patches and
  the loss is computed at the bag level, so the model is trained end to end for
  patient-level survival classification.

## Setup

The project runs in the conda environment named `base`. PyTorch
(`torch 2.6.0+cu124`) is already installed there. The remaining Python
dependencies (`timm`, `albumentations`, `wandb`, and the rest) come from
`requirements.txt`:

```bash
conda activate base
pip install -r requirements.txt
```

Two environment notes that matter on machines without outbound network access:

- **Pretrained weights are loaded offline.** `timm` is told to resolve the
  EfficientNet-B2 ImageNet weights from the torch hub cache rather than the
  Hugging Face Hub. Place `efficientnet_b2_ra-bcdf34b7.pth` in the torch hub
  cache (`~/.cache/torch/hub/checkpoints/` on Linux, the equivalent under the
  user profile on Windows) before the first run.
- **Weights & Biases runs in offline mode.** Set `WANDB_MODE=offline` to keep
  logging local; runs are written under `artifacts/wandb/`. If `WANDB_API_KEY`
  is unset and offline mode is not configured, training continues without
  logging.

## Run

All commands are run from the project root.

```bash
# BreaKHis — train the patch-level classifier
python src/training/train.py --config configs/breakhis.yaml

# BreaKHis — evaluate a checkpoint on the test split
python src/training/evaluate.py \
    --config configs/breakhis.yaml \
    --checkpoint artifacts/results/checkpoints/best.pth

# DLBCL — patch-level baseline
python src/training/train.py --config configs/dlbcl.yaml

# DLBCL — end-to-end attention MIL (ABMIL)
python src/training/train_mil.py --config configs/dlbcl_mil.yaml
```

`train_mil.py` selects the best checkpoint by validation patient-level AUC, then
evaluates that checkpoint on the test split and appends a results block to
`DLBCL_MIL_RESULT.md`.

## Data

The image data is large and is not tracked in git. Download and unpack it under
`data/`:

- **BreaKHis** from the official BreaKHis release.
- **DLBCL** from TCIA (DLBC-Morphology collection).

The split CSV files are tracked in git under `data/splits/`. The DLBCL survival
split is produced by `scripts/build_survival_split.py` (seed = 42), which reads
overall-survival (OS) metadata from the clinical CSV and applies a 5-year OS rule:
label `1` = death within 5 years (event and OS < 5y), label `0` = survival beyond
5 years (OS >= 5y); patients censored before 5 years are dropped as unlabelable.
The script enforces patient-level separation between
train, validation, and test so no patient appears in more than one split.

## Datasets

| Dataset | Magnification | Images / patches | Split (patient-level) |
|---|---|---|---|
| BreaKHis | 40x | ~1,995 images, 82 patients | train 1,330 / val 295 / test 370 |
| DLBCL survival | — | see below | train 71 / val 15 / test 16 patients |

**BreaKHis.** Only 40x images are used. The split is stratified at the patient
level (70/15/15) so that no patient crosses split boundaries.

**DLBCL survival.** The cohort starts from 170 patients; 102 of them have a
usable 5-year OS label (the rest are censored before 5 years and dropped). These 102 patients
are split patient-wise into 71 train / 15 validation / 16 test, giving
11,334 / 2,668 / 3,072 patches respectively. Labels are patient-level and shared
by every patch of a patient.

The DLBCL patch dataset filters missing image files during initialisation: any
CSV row whose file is absent on disk is dropped before training, and the patient
index is rebuilt from the rows that remain.

## Results

| Setting | Test metric | Value |
|---|---|---|
| BreaKHis (image-level) | AUC | 0.954 |
| DLBCL ABMIL (patient-level) | AUC | 0.509 |

BreaKHis reaches a test AUC of 0.954. The DLBCL ABMIL model reaches a
patient-level test AUC of 0.509, below the 0.75 target. With only 71 training
patients and 16 test patients the bag-level model overfits and does not
generalise; this is a dataset-size limitation rather than a pipeline defect, and
is reported here as-is.

## Repository layout

See [architecture.md](architecture.md) for the file tree and a description of
each module.
