#!/usr/bin/env bash
# run_ablation.sh — Run ablation experiments (#017 and #018)
#
# Usage:
#   bash run_ablation.sh                 # run both ablations
#   bash run_ablation.sh --study breakhis   # #017 only
#   bash run_ablation.sh --study dlbcl     # #018 only
#
# Outputs:
#   results/ablation_breakhis.json   (#017)
#   results/dlbcl_eval.json          (#018, three lr runs)
#   results/ablation_dlbcl.json      (#018 summary)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

STUDY="both"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --study) STUDY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

CONDA="conda run -n ai_dt --no-capture-output python"
RESULTS_DIR="results/checkpoints"
mkdir -p "$RESULTS_DIR" results/

# ── #017: BreaKHis augmentation ablation ─────────────────────────────────────
ablation_breakhis() {
  echo ""
  echo "========================================"
  echo "  #017: BreaKHis Augmentation Ablation"
  echo "========================================"

  # Run 1: WITH augmentation (standard config)
  echo "[1/2] Training WITH augmentation..."
  $CONDA src/training/train.py --config configs/breakhis.yaml
  BEST_AUG="$RESULTS_DIR/best.pth"

  $CONDA src/training/evaluate.py \
    --config configs/breakhis.yaml \
    --checkpoint "$BEST_AUG"
  AUC_AUG=$(python -c "import json; d=json.load(open('results/breakhis_eval.json')); print(d['metrics']['auc'])")
  echo "  AUC (with aug): $AUC_AUG"

  # Save with-aug checkpoint
  cp "$BEST_AUG" "$RESULTS_DIR/best_breakhis_aug.pth"

  # Run 2: WITHOUT augmentation (override config)
  echo "[2/2] Training WITHOUT augmentation..."
  python - <<PYEOF
import yaml, copy, tempfile, os, subprocess, sys

with open("configs/breakhis.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["augmentation"] = False  # signal to disable aug
cfg["wandb"]["project"] = "cancer-histo-breakhis-noaug"

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.dump(cfg, tmp)
tmp.close()
print(f"No-aug config: {tmp.name}")

ret = subprocess.run(
    ["conda", "run", "-n", "ai_dt", "--no-capture-output",
     "python", "src/training/train.py", "--config", tmp.name],
    check=False
)
os.unlink(tmp.name)
sys.exit(ret.returncode)
PYEOF

  BEST_NOAUG="$RESULTS_DIR/best.pth"
  # Temporarily rename to avoid overwrite
  cp "$BEST_NOAUG" "$RESULTS_DIR/best_breakhis_noaug.pth"

  python - <<PYEOF
import yaml, tempfile, os, json
with open("configs/breakhis.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["augmentation"] = False
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.dump(cfg, tmp); tmp.close()

import subprocess, sys
ret = subprocess.run(
    ["conda", "run", "-n", "ai_dt", "--no-capture-output",
     "python", "src/training/evaluate.py",
     "--config", tmp.name,
     "--checkpoint", "results/checkpoints/best_breakhis_noaug.pth"],
    check=False
)
os.unlink(tmp.name)
sys.exit(ret.returncode)
PYEOF

  AUC_NOAUG=$(python -c "import json; d=json.load(open('results/breakhis_eval.json')); print(d['metrics']['auc'])")
  echo "  AUC (no aug): $AUC_NOAUG"

  # Write ablation summary
  python - <<PYEOF
import json
from datetime import datetime, timezone

result = {
    "study": "breakhis_augmentation",
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": {
        "with_augmentation": {"auc": $AUC_AUG},
        "without_augmentation": {"auc": $AUC_NOAUG},
    }
}
with open("results/ablation_breakhis.json", "w") as f:
    json.dump(result, f, indent=2)
print("Written: results/ablation_breakhis.json")
print(json.dumps(result["runs"], indent=2))
PYEOF
}

# ── #018: DLBCL learning-rate ablation ───────────────────────────────────────
ablation_dlbcl() {
  echo ""
  echo "========================================"
  echo "  #018: DLBCL Learning-Rate Ablation"
  echo "========================================"

  declare -A LR_AUCS

  for LR in 1e-3 1e-4 1e-5; do
    echo ""
    echo "--- Training with lr=$LR ---"

    python - <<PYEOF
import yaml, tempfile, os, subprocess, sys

with open("configs/dlbcl.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["lr"] = float("$LR")
cfg["wandb"]["project"] = f"cancer-histo-dlbcl-lr{\"$LR\".replace(\"-\", \"\")}"

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.dump(cfg, tmp); tmp.close()

ret = subprocess.run(
    ["conda", "run", "-n", "ai_dt", "--no-capture-output",
     "python", "src/training/train.py", "--config", tmp.name],
    check=False
)
os.unlink(tmp.name)
sys.exit(ret.returncode)
PYEOF

    CKPT="$RESULTS_DIR/best_dlbcl.pth"
    python - <<PYEOF
import yaml, tempfile, os, subprocess, sys

with open("configs/dlbcl.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["lr"] = float("$LR")
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.dump(cfg, tmp); tmp.close()

ret = subprocess.run(
    ["conda", "run", "-n", "ai_dt", "--no-capture-output",
     "python", "src/training/evaluate.py",
     "--config", tmp.name,
     "--checkpoint", "$CKPT"],
    check=False
)
os.unlink(tmp.name)
sys.exit(ret.returncode)
PYEOF

    AUC=$(python -c "import json; d=json.load(open('results/dlbcl_eval.json')); print(d['metrics']['auc'])")
    echo "  lr=$LR → AUC=$AUC"
    LR_AUCS[$LR]=$AUC

    # Save per-lr checkpoint
    cp "$CKPT" "$RESULTS_DIR/best_dlbcl_lr${LR}.pth"
  done

  # Write ablation summary
  python - <<PYEOF
import json
from datetime import datetime, timezone

result = {
    "study": "dlbcl_learning_rate",
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": {
        "lr_1e-3": {"lr": 1e-3, "auc": ${LR_AUCS[1e-3]:-0}},
        "lr_1e-4": {"lr": 1e-4, "auc": ${LR_AUCS[1e-4]:-0}},
        "lr_1e-5": {"lr": 1e-5, "auc": ${LR_AUCS[1e-5]:-0}},
    }
}
with open("results/ablation_dlbcl.json", "w") as f:
    json.dump(result, f, indent=2)
print("Written: results/ablation_dlbcl.json")
print(json.dumps(result["runs"], indent=2))
PYEOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$STUDY" in
  both)
    ablation_breakhis
    ablation_dlbcl
    ;;
  breakhis)
    ablation_breakhis
    ;;
  dlbcl)
    ablation_dlbcl
    ;;
  *)
    echo "Unknown study: $STUDY"
    exit 1
    ;;
esac

echo ""
echo "=== run_ablation.sh complete ==="
echo "Outputs:"
ls -lh results/ablation_*.json 2>/dev/null || true
