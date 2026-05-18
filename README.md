# Cancer Histology Classification

Binary classification of histological images using EfficientNet-B2, evaluated on **BreaKHis** (breast cancer, 40x) and **DLBCL** (lymphoma) datasets.

## Setup

```bash
conda create -n cancer-histo python=3.10 -y
conda activate cancer-histo
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run

```bash
# Train
python src/training/train.py --config configs/breakhis.yaml

# Evaluate
python src/training/evaluate.py --config configs/breakhis.yaml --checkpoint results/checkpoints/best.pth
```
