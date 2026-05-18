# PRD — Cancer Histopathology Classification & Segmentation

## 1. 功能目标

用深度学习自动分析病理切片图像，对 BreaKHis 数据集做良恶性分类，对 DLBCL 数据集做患者生存预测，并通过 GradCAM 热力图实现分类决策的可视化解释。

## 2. 用户场景

- **谁**：6 人硕士研究团队（2 人写代码 / 2 人跑实验 / 2 人写报告）
- **情境**：在课程项目周期内，使用各自本地 GPU（4060 Laptop 8GB）和 GitHub 协作
- **为什么**：完成课程要求的分类+分割工程实现，产出可演示的实验结果用于期末答辩评分

## 3. 关键接口

**Dataset 统一返回格式（所有子类必须遵守）：**
```python
{"image": Tensor(3,224,224), "label": int, "patient_id": str, "meta": dict}
```

**配置文件（YAML）关键字段：**
```yaml
model: efficientnet_b2
magnification: 40x        # BreaKHis only
batch_size: 32
lr: 1e-4
epochs: 30
amp: true                 # 混合精度
```

**W&B 项目名**：`cancer-histo-{breakhis|dlbcl}`，所有成员使用同一 entity。

## 4. 验收标准

- [ ] `python src/datasets/breakhis_dataset.py` 无报错，打印 sample shape `(3, 224, 224)` 和 label `0/1`
- [ ] `python train.py --config configs/breakhis.yaml` 在 4060 Laptop 上单 epoch 耗时 < 5 分钟
- [ ] BreaKHis 测试集 AUC ≥ 0.85（40× 倍数）
- [ ] DLBCL 测试集 AUC ≥ 0.75（patient-level 聚合后）
- [ ] `python gradcam.py --image <path>` 输出热力图 PNG，非全黑/全白
- [ ] `evaluate.py` 输出包含 Accuracy / AUC / F1 / 混淆矩阵的 JSON 文件
- [ ] GitHub 仓库包含 `requirements.txt`，`pip install -r requirements.txt` 无冲突

## 5. 不在本次范围内

- 像素级语义分割（无 ground truth mask，不做有监督分割训练）
- 多倍数 ensemble（仅使用 40× 单倍数）
- 多卡 / 分布式训练
- 模型部署（API 服务 / Docker）
- 报告正文写作（由团队成员自行完成，禁止直接使用 AI 生成文本）
- 超过 BreaKHis + DLBCL 的第三方数据集引入

---

**自检：**
- [x] 字数 < 500（正文约 280 字）
- [x] 验收标准全部可测试（命令行可运行 / 数值可量化）
- [x] 范围外事项已显式列出（6 条）
- [x] 一句话目标普通人能看懂
