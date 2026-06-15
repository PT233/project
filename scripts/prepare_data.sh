#!/usr/bin/env bash
# prepare_data.sh — Download datasets and generate CSV splits
#
# Usage:
#   bash scripts/prepare_data.sh [--skip-download]   # skip if already downloaded
#
# After completion:
#   data/breakhis/  — BreaKHis images (40x only used)
#   data/dlbcl/     — DLBCL patch images
#   data/splits/    — train.csv / val.csv / test.csv for each dataset

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BREAKHIS_URL="https://nextcloud.univ-lille.fr/index.php/s/TXHR4qjo6PFamJC/download"
DLBCL_URL="https://nextcloud.univ-lille.fr/index.php/s/CnGD56yX9WZtcAX/download"
SPLITS_URL="https://nextcloud.univ-lille.fr/index.php/s/dfdbEA3npkcB9oo/download"

SKIP_DOWNLOAD=0
for arg in "$@"; do
  [[ "$arg" == "--skip-download" ]] && SKIP_DOWNLOAD=1
done

mkdir -p data/breakhis data/dlbcl data/splits

# ── 1. Download  
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  echo "[1/3] Downloading BreaKHis..."
  wget -q --show-progress -O /tmp/breakhis.zip "$BREAKHIS_URL"
  echo "[1/3] Extracting BreaKHis..."
  unzip -q /tmp/breakhis.zip -d data/breakhis/
  rm /tmp/breakhis.zip

  echo "[2/3] Downloading DLBCL..."
  wget -q --show-progress -O /tmp/dlbcl.zip "$DLBCL_URL"
  echo "[2/3] Extracting DLBCL..."
  unzip -q /tmp/dlbcl.zip -d data/dlbcl/
  rm /tmp/dlbcl.zip

  echo "[3/3] Downloading CSV splits..."
  wget -q --show-progress -O /tmp/splits.zip "$SPLITS_URL"
  echo "[3/3] Extracting splits..."
  unzip -q /tmp/splits.zip -d data/splits/
  rm /tmp/splits.zip
else
  echo "[skip] Download skipped (--skip-download)"
fi

# ── 2. Inspect downloaded splits 
echo ""
echo " ' data/splits/ contents  '"
ls -lh data/splits/

# ── 3. Normalise CSV names to what train.py expects 
# train.py  expects: data/splits/train.csv, data/splits/val.csv
# evaluate.py checks: {dataset}_test.csv → {dataset}_val.csv → test.csv → val.csv
# Strategy: if provider gives breakhis_train.csv etc, create symlinks to train.csv
echo ""
echo " ' Normalising CSV filenames  '"
cd data/splits/

normalize_csv() {
  local prefix="$1"   # e.g. "breakhis" or "dlbcl"
  for split in train val test; do
    src="${prefix}_${split}.csv"
    dst="${split}.csv"
    if [[ -f "$src" && ! -f "$dst" ]]; then
      ln -sf "$src" "$dst"
      echo "  linked $src → $dst"
    fi
  done
}

# Try both prefixes; if plain train.csv already exists, nothing to do
normalize_csv "breakhis"
normalize_csv "dlbcl"

# Verify at least train.csv and val.csv exist
for f in train.csv val.csv; do
  if [[ ! -f "$f" ]]; then
    echo "[WARN] $f still not found — you may need to rename manually"
  else
    echo "  OK: $f ($(wc -l < "$f") lines)"
  fi
done

cd "$PROJECT_ROOT"

# ── 4. Quick sanity check  
echo ""
echo " ' Sanity check  '"
conda run -n cancer-histo python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from src.datasets.breakhis_dataset import BreaKHisDataset
import glob

csv_candidates = [
    "data/splits/train.csv",
    "data/splits/breakhis_train.csv",
]
csv_path = next((c for c in csv_candidates if os.path.exists(c)), None)
if csv_path is None:
    print("[WARN] No BreaKHis train CSV found — skipping BreaKHis check")
else:
    ds = BreaKHisDataset(csv_path=csv_path, data_root="data/breakhis/", mode="val")
    print(f"BreaKHis dataset size: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"  image shape: {sample['image'].shape}")
        print(f"  label: {sample['label']}, patient_id: {sample['patient_id']}")
PYEOF

echo ""
echo " ' prepare_data.sh complete  '"
echo "Next: bash scripts/run_training.sh"
