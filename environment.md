# environment.md — 运行环境

> 模块结构见 [architecture.md](architecture.md)，命令参数见 [protocol.md](protocol.md)

## 1. 运行时依赖

### 硬件要求

| 项目 | 最低 | 推荐（本项目实际） |
|------|------|-------------------|
| GPU | 6GB VRAM | NVIDIA RTX 4060 Laptop 8GB |
| RAM | 16GB | 16GB+ |
| 存储 | 10GB 可用空间 | 20GB（BreaKHis ~2.3GB + DLBCL ~未知） |
| CUDA | 11.8+ | 12.1 |

### 软件依赖（固定版本）

```
python==3.10.*
torch==2.2.2+cu121
torchvision==0.17.2+cu121
timm==0.9.16
opencv-python==4.9.0.80
albumentations==1.4.3
grad-cam==1.4.8
pyyaml==6.0.1
pandas==2.2.1
scikit-learn==1.4.1
wandb==0.16.6
matplotlib==3.8.4
seaborn==0.13.2
tqdm==4.66.2
Pillow==10.3.0
```

> 完整列表见 `requirements.txt`，使用 `pip install -r requirements.txt` 安装。

### Conda 环境（推荐）

```bash
conda create -n cancer-histo python=3.10 -y
conda activate cancer-histo
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. 环境变量与配置文件

### 环境变量

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `WANDB_API_KEY` | ✅ | W&B 个人 API Key，从 wandb.ai/settings 获取 |
| `WANDB_ENTITY` | ✅ | 团队共享 entity 名称（6人统一） |
| `DATA_ROOT` | ❌ | 覆盖 YAML 中的 `data_root`，用于不同机器路径不同的场景 |

在项目根目录创建 `.env` 文件（**不得提交到 git**）：

```bash
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_ENTITY=your-team-entity
```

### 配置文件位置

| 文件 | 用途 | 谁修改 |
|------|------|--------|
| `configs/breakhis.yaml` | BreaKHis 超参数 | Human C |
| `configs/dlbcl.yaml` | DLBCL 超参数 | Human D |
| `.env` | 密钥（本地，不入git） | 每人自己配置 |
| `.gitignore` | 排除数据/结果/密钥 | Human A 维护 |

## 3. 部署方式

**本项目仅支持本地运行**，无 Docker，无云部署（在 PRD 范围外）。

每位成员在自己机器上独立运行训练，通过 W&B dashboard 共享实验结果。

数据集同步方式：
1. 从课程提供的 Nextcloud 链接下载 BreaKHis 和 DLBCL
2. 从 Nextcloud 下载 CSV splits
3. 按 `architecture.md` 的目录结构放置到 `data/` 下
4. `data/` 目录**不推送到 GitHub**（`.gitignore` 已排除）

## 4. 关键路径与权限

| 路径 | 读/写 | 说明 |
|------|-------|------|
| `data/` | 读 | 原始数据，训练时只读 |
| `data/splits/` | 读 | CSV文件，git托管 |
| `results/checkpoints/` | 写 | 自动创建，存储 `.pth` 文件 |
| `results/gradcam/` | 写 | 自动创建，存储热力图 PNG |
| `~/.cache/torch/hub/` | 读/写 | timm 自动下载预训练权重 |
| `.env` | 读 | 训练启动时由 `logger.py` 自动加载 |

无特殊系统权限要求，普通用户权限即可运行。

## 5. 启停命令与健康检查

### 环境验证（首次配置后运行）

```bash
# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 验证数据集可读
python src/datasets/breakhis_dataset.py

# 验证 W&B 连接
python -c "import wandb; wandb.login()"
```

预期输出：
```
True  NVIDIA GeForce RTX 4060 Laptop GPU
Sample shape: torch.Size([3, 224, 224]), label: 0
wandb: Successfully logged in.
```

### 训练启动

```bash
conda activate cancer-histo
python src/training/train.py --config configs/breakhis.yaml
```

### 中断与恢复

训练会在每个 epoch 结束时保存 `results/checkpoints/last.pth`，中断后使用：

```bash
python src/training/train.py --config configs/breakhis.yaml --resume results/checkpoints/last.pth
```

### 评估

```bash
python src/training/evaluate.py --config configs/breakhis.yaml --checkpoint results/checkpoints/best.pth
# 输出: results/breakhis_eval.json
```

### 健康检查指标

| 检查项 | 预期值 | 异常处理 |
|--------|--------|----------|
| 单 epoch 时间 | < 5 分钟 | 超出则降低 `batch_size` 或检查数据加载 `num_workers` |
| GPU 显存占用 | < 7.5GB | 超出则开启 `amp: true` 或降低 `batch_size` |
| val AUC（epoch 10） | > 0.75 | 低于则检查学习率和数据标准化 |
| W&B 日志延迟 | < 30s | 仅警告，不中断训练 |
