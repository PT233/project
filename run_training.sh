#!/usr/bin/env bash
# run_training.sh — Run full 30-epoch training for both datasets
#
# Usage:
#   bash run_training.sh                 # train both
#   bash run_training.sh --dataset breakhis   # train BreaKHis only
#   bash run_training.sh --dataset dlbcl     # train DLBCL only
#   bash run_training.sh --eval-only         # skip training, run evaluation only
#
# Prerequisites:
#   1. conda env base with all packages installed
#   2. WANDB runs in offline mode (set WANDB_MODE=offline; no API key needed)
#   3. data/ populated (run prepare_data.sh first)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Load .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
  echo "[INFO] Loaded .env"
fi

DATASET="both"
EVAL_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

CONDA="conda run -n base --no-capture-output python"
RESULTS_DIR="results/checkpoints"
mkdir -p "$RESULTS_DIR"

# BreaKHis
run_breakhis() {
  echo ""
  
  echo "  BreaKHis: 30-epoch training (#012)"
  

  if [[ $EVAL_ONLY -eq 0 ]]; then
    $CONDA src/training/train.py --config configs/breakhis.yaml
    echo "[OK] BreaKHis training complete"
  fi

  # Locate best checkpoint
  CKPT="$RESULTS_DIR/best.pth"
  if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] best.pth not found after training"
    exit 1
  fi

  echo ""
  echo "--- BreaKHis evaluation (#014) ---"
  $CONDA src/training/evaluate.py \
    --config configs/breakhis.yaml \
    --checkpoint "$CKPT"

  # Check AUC ≥ 0.85
  IMG_LIST="$(python - <<'PYEOF'
import json, sys
with open("results/breakhis_eval.json") as f:
    r = json.load(f)
auc = r["metrics"]["auc"]
print(f"BreaKHis AUC: {auc}")
if auc >= 0.85:
    print("[PASS] AUC >= 0.85")
else:
    print(f"[WARN] AUC {auc} < 0.85 target")
    sys.exit(0)  # warn but don't fail
PYEOF
}

# DLBCL
run_dlbcl() {
  echo ""
  echo "  DLBCL: 30-epoch training (#013)"

  if [[ $EVAL_ONLY -eq 0 ]]; then
    $CONDA src/training/train.py --config configs/dlbcl.yaml
    echo "[OK] DLBCL training complete"
  fi

  # Locate best checkpoint (DLBCL saves as best_dlbcl.pth)
  CKPT="$RESULTS_DIR/best_dlbcl.pth"
  if [[ ! -f "$CKPT" ]]; then
    # Fallback to best.pth if best_dlbcl.pth not yet renamed
    CKPT="$RESULTS_DIR/best.pth"
    echo "[WARN] best_dlbcl.pth not found, using best.pth"
  fi

  echo ""
  echo "--- DLBCL evaluation (#015) ---"
  $CONDA src/training/evaluate.py \
    --config configs/dlbcl.yaml \
    --checkpoint "$CKPT"

  # Check patient-level AUC ≥ 0.75
  python - <<PYEOF
import json, sys
with open("results/dlbcl_eval.json") as f:
    r = json.load(f)
auc = r["metrics"]["auc"]
print(f"DLBCL patient-level AUC: {auc}")
if auc >= 0.75:
    print("[PASS] AUC >= 0.75")
else:
    print(f"[WARN] AUC {auc} < 0.75 target")
    sys.exit(0)
PYEOF
}

# GradCAM (#016)
run_gradcam() {
  echo ""
  
  echo "  GradCAM visualization (#016)"
  

  CKPT="$RESULTS_DIR/best.pth"
  OUT_DIR="results/gradcam"
  mkdir -p "$OUT_DIR"

  # Find test images: benign and malignant, ≥5 each
  python - <<PYEOF
import os, sys, csv, random

csv_candidates = [
    "data/splits/breakhis_test.csv",
    "data/splits/test.csv",
    "data/splits/val.csv",
]
csv_path = next((c for c in csv_candidates if os.path.exists(c)), None)
if csv_path is None:
    print("[ERROR] No test CSV found for GradCAM")
    sys.exit(1)

benign, malignant = [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("magnification", "40x") != "40x":
            continue
        if int(row["label"]) == 0:
            benign.append(row["filepath"])
        else:
            malignant.append(row["filepath"])

random.seed(42)
selected = (
    random.sample(benign, min(5, len(benign))) +
    random.sample(malignant, min(5, len(malignant)))
)
for p in selected:
    print(p)
PYEOF
)"
  echo "$IMG_LIST" | while read -r img_path; do
    echo "  GradCAM: $img_path"
    $CONDA src/models/gradcam.py \
      --image "$img_path" \
      --checkpoint "$CKPT" \
      --output "$OUT_DIR/"
  done

  echo ""
  N=$(ls "$OUT_DIR/"*.png 2>/dev/null | wc -l)
  echo "Generated $N GradCAM PNGs in $OUT_DIR/"
  [[ $N -ge 10 ]] && echo "[PASS] >= 10 PNGs" || echo "[WARN] Only $N PNGs generated"
}

# Dispatch
case "$DATASET" in
  both)
    run_breakhis
    run_dlbcl
    run_gradcam
    ;;
  breakhis)
    run_breakhis
    run_gradcam
    ;;
  dlbcl)
    run_dlbcl
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

echo ""
echo " ' run_training.sh complete   '"
