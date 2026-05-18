# protocol.md — 接口与协议

> 模块归属见 [architecture.md](architecture.md)，运行命令见 [environment.md](environment.md)

## 1. 对外接口（命令行）

本项目无 HTTP API，所有对外接口均为命令行调用。

### train.py

```bash
python src/training/train.py --config configs/breakhis.yaml [--resume results/checkpoints/last.pth]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--config` | str | ✅ | YAML配置文件路径 |
| `--resume` | str | ❌ | checkpoint路径，从断点继续训练 |

退出码：`0` 成功，`1` 配置错误，`2` 数据路径不存在，`3` 显存OOM

### evaluate.py

```bash
python src/training/evaluate.py --config configs/breakhis.yaml --checkpoint results/checkpoints/best.pth
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--config` | str | ✅ | 需与训练时一致 |
| `--checkpoint` | str | ✅ | 模型权重路径 |

输出：`results/{dataset}_eval.json`（格式见第3节）

### gradcam.py

```bash
python src/models/gradcam.py --image data/breakhis/sample.png --checkpoint results/checkpoints/best.pth --output results/gradcam/
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--image` | str | ✅ | 单张图像路径 |
| `--checkpoint` | str | ✅ | 模型权重路径 |
| `--output` | str | ❌ | 输出目录，默认 `results/gradcam/` |

## 2. 内部通信协议

### Dataset 统一返回格式

所有 `Dataset.__getitem__` 必须返回以下字典，**字段名和类型不得变更**：

```python
{
    "image":      torch.Tensor,  # shape: (3, 224, 224), dtype: float32, 已归一化
    "label":      int,           # 0 = benign/high-survival, 1 = malignant/low-survival
    "patient_id": str,           # BreaKHis: SOB_B_A-14-22549G; DLBCL: patient_001
    "meta":       dict           # 当前为空 {}，预留扩展
}
```

### DLBCL Patient-level 聚合协议

`evaluate.py` 对 DLBCL 执行以下聚合：

```
patch_probs: List[float]  # 同一patient_id的所有patch预测概率
patient_prob = max(patch_probs)
patient_pred = 1 if patient_prob >= 0.5 else 0
```

### Checkpoint 格式

```python
{
    "epoch":       int,
    "model_state": OrderedDict,   # model.state_dict()
    "optim_state": OrderedDict,   # optimizer.state_dict()
    "best_auc":    float,
    "config":      dict           # 训练时的完整YAML内容
}
```

## 3. 数据格式规范

### YAML 配置文件结构

```yaml
dataset: breakhis          # 枚举: breakhis | dlbcl
data_root: data/breakhis/
split_dir: data/splits/
magnification: 40x         # BreaKHis only，dlbcl配置忽略此字段

model: efficientnet_b2
pretrained: true
num_classes: 2

batch_size: 32
lr: 1.0e-4
weight_decay: 1.0e-5
epochs: 30
amp: true

wandb:
  project: cancer-histo-breakhis   # 命名规范: cancer-histo-{dataset}
  entity: YOUR_WANDB_ENTITY        # 6人使用同一entity
```

### 评估输出 JSON（`results/{dataset}_eval.json`）

```json
{
  "dataset": "breakhis",
  "checkpoint": "results/checkpoints/best.pth",
  "timestamp": "2026-05-18T14:30:00Z",
  "metrics": {
    "accuracy": 0.912,
    "auc": 0.957,
    "f1": 0.903,
    "confusion_matrix": [[210, 18], [15, 489]]
  }
}
```

字段规范：
- 时间戳统一使用 ISO 8601 UTC 格式
- 浮点指标保留 3 位小数
- `confusion_matrix` 行为真实标签，列为预测标签，顺序 `[[TN, FP], [FN, TP]]`

### CSV Split 文件格式

```
filepath,label,patient_id,magnification
data/breakhis/SOB_B_A-14-22549G-40-001.png,0,SOB_B_A-14-22549G,40x
```

- `filepath`：相对于项目根目录的路径
- `label`：整数 0 或 1
- `magnification`：DLBCL 此列填 `N/A`

## 4. 错误码与异常处理

| 退出码 | 含义 | 处理方式 |
|--------|------|----------|
| `0` | 成功 | — |
| `1` | 配置文件缺失或字段非法 | 打印缺失字段名后退出 |
| `2` | 数据路径不存在 | 打印期望路径后退出 |
| `3` | CUDA OOM | 提示降低 `batch_size` 后退出 |
| `4` | Checkpoint 版本不匹配 | 打印 checkpoint 内嵌 config 差异后退出 |

所有异常统一通过 `logger.py` 输出，格式：
```
[ERROR][{module}] {message}
```

不使用 `print()` 输出错误信息，统一用 Python `logging` 模块。

## 5. 版本兼容与约束

- Checkpoint 跨版本不兼容：`config.model` 字段变更时旧 checkpoint 拒绝加载（退出码4）
- YAML 新增字段向后兼容（旧字段缺失时使用默认值），**删除字段不兼容**
- Dataset 返回字典新增 `meta` 子字段向后兼容，修改已有字段类型不兼容
- 无网络超时设置（W&B 日志失败时仅打印警告，不中断训练）
