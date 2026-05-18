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

## Dataset Integrity Notes

### BreaKHis

| 项目 | 值 |
|------|----|
| 验证状态 | ✅ zip 完整（`No errors detected`） |
| 40x 图像数 | 1,995 张（benign=625, malignant=1,370）|
| 患者数 | 82 |
| Split（患者级别分层） | train=1,330 / val=295 / test=370 |
| Split 文件 | `data/splits/breakhis_{train,val,test}.csv` |

> **说明**：官方提供的 CSV（`CSV/breakhis_train.csv`）只含 200X/400X，与 protocol.md 的 40X 约定不符。已重新从磁盘图像生成 40X 专用 split，按患者级别分层（70/15/15）防止数据泄露。

### DLBCL

| 项目 | 值 |
|------|----|
| 验证状态 | ⚠️ 部分缺失（已过滤处理）|
| CSV 原始总 patch 数 | 26,873 |
| 磁盘实际存在 | 24,590（91.5%）|
| 缺失 patch 数 | 2,283（8.5%，分散于全部 153 个患者）|
| 平均每患者缺失率 | 10.5%（最高：患者 26802，63.6%）|
| 实际使用 patch 数 | train=19,424 / val=5,166（已过滤缺失）|
| 患者数 | train=121 / val=32 |
| Split 文件 | `data/splits/dlbcl_{train,val}.csv` |

> **处理方式**：重新生成的 CSV split（`dlbcl_train.csv` / `dlbcl_val.csv`）已在生成时过滤掉缺失文件，仅保留磁盘实际存在的 patch。`DLBCLDataset` 对缺失文件的处理也已从 `sys.exit(2)` 改为返回零图像（warning 级日志），防止训练中途崩溃。
