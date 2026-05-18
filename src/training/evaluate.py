"""
evaluate.py — Model evaluation script.

Loads a trained checkpoint and evaluates it on the test split.

Usage:
    python src/training/evaluate.py --config configs/breakhis.yaml --checkpoint results/checkpoints/best.pth

Exit codes:
    0 — success
    1 — config error (missing file or invalid fields)
    2 — data path does not exist
    4 — checkpoint model mismatch with config
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import torch
import yaml
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Module-level logger — format: [ERROR][evaluate] <message>
# ---------------------------------------------------------------------------
_log = logging.getLogger("evaluate")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
_log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False


# ---------------------------------------------------------------------------
# Helper: load YAML config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate YAML config file.

    Returns parsed config dict.
    Exits with code 1 on any config error.
    """
    if not os.path.exists(config_path):
        _log.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            _log.error("Failed to parse config YAML: %s", exc)
            sys.exit(1)

    required_fields = ["dataset", "data_root", "split_dir", "model", "num_classes"]
    missing = [k for k in required_fields if k not in cfg]
    if missing:
        _log.error("Config missing required fields: %s", missing)
        sys.exit(1)

    if cfg["dataset"] not in ("breakhis", "dlbcl"):
        _log.error("Config field 'dataset' must be 'breakhis' or 'dlbcl', got: %s", cfg["dataset"])
        sys.exit(1)

    return cfg


# ---------------------------------------------------------------------------
# Helper: load checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, cfg: dict) -> "OrderedDict":
    """Load model_state from checkpoint.

    Validates that checkpoint config.model matches current config.model.
    Exits with code 4 on model mismatch.

    Returns the model_state OrderedDict.
    """
    if not os.path.exists(ckpt_path):
        _log.error("Checkpoint file not found: %s", ckpt_path)
        sys.exit(1)

    device = "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    # Validate checkpoint format
    if "model_state" not in ckpt:
        _log.error("Checkpoint missing 'model_state' field: %s", ckpt_path)
        sys.exit(4)

    # Check model compatibility if checkpoint embeds config
    if "config" in ckpt and ckpt["config"] is not None:
        ckpt_model = ckpt["config"].get("model", None)
        cfg_model = cfg.get("model", None)
        if ckpt_model is not None and cfg_model is not None and ckpt_model != cfg_model:
            _log.error(
                "Checkpoint config.model='%s' does not match current config.model='%s'",
                ckpt_model,
                cfg_model,
            )
            sys.exit(4)

    return ckpt["model_state"]


# ---------------------------------------------------------------------------
# Helper: build dataset
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict):
    """Instantiate the correct dataset class for the val/test split.

    Exits with code 2 if data paths do not exist.
    Returns the dataset instance.
    """
    dataset_name = cfg["dataset"]
    split_dir = cfg["split_dir"]
    data_root = cfg["data_root"]

    if not os.path.exists(split_dir):
        _log.error("split_dir does not exist: %s", split_dir)
        sys.exit(2)

    if not os.path.exists(data_root):
        _log.error("data_root does not exist: %s", data_root)
        sys.exit(2)

    # Resolve test/val CSV path
    # Convention: {split_dir}/{dataset}_test.csv or {split_dir}/{dataset}_val.csv
    csv_candidates = [
        os.path.join(split_dir, f"{dataset_name}_test.csv"),
        os.path.join(split_dir, f"{dataset_name}_val.csv"),
        os.path.join(split_dir, "test.csv"),
        os.path.join(split_dir, "val.csv"),
    ]
    csv_path = None
    for candidate in csv_candidates:
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if csv_path is None:
        _log.error(
            "No test/val CSV found in split_dir='%s'. Tried: %s",
            split_dir,
            csv_candidates,
        )
        sys.exit(2)

    _log.info("Using split CSV: %s", csv_path)

    if dataset_name == "breakhis":
        from src.datasets.breakhis_dataset import BreaKHisDataset
        dataset = BreaKHisDataset(csv_path=csv_path, data_root=data_root, mode="val")
    else:  # dlbcl
        from src.datasets.dlbcl_dataset import DLBCLDataset
        dataset = DLBCLDataset(csv_path=csv_path, data_root=data_root, mode="val")

    return dataset


# ---------------------------------------------------------------------------
# Helper: run inference
# ---------------------------------------------------------------------------

def run_inference(model, dataset, batch_size: int = 32):
    """Run model inference on all samples in dataset.

    Returns:
        all_probs  : list of float — softmax probability for class 1
        all_labels : list of int   — ground-truth labels
        all_pids   : list of str   — patient_id for each sample
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_probs = []
    all_labels = []
    all_pids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            pids = batch["patient_id"]

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

            all_probs.extend(probs)
            all_labels.extend(labels.tolist() if hasattr(labels, "tolist") else list(labels))
            all_pids.extend(list(pids))

    return all_probs, all_labels, all_pids


# ---------------------------------------------------------------------------
# Helper: compute metrics
# ---------------------------------------------------------------------------

def compute_image_level_metrics(probs, labels):
    """Compute image-level evaluation metrics.

    Returns dict with accuracy, auc, f1, confusion_matrix.
    """
    preds = [1 if p >= 0.5 else 0 for p in probs]

    accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds, average="weighted")
    cm = confusion_matrix(labels, preds).tolist()

    return {
        "accuracy": round(accuracy, 3),
        "auc": round(auc, 3),
        "f1": round(f1, 3),
        "confusion_matrix": cm,
    }


def compute_patient_level_metrics(probs, labels, pids):
    """Compute patient-level evaluation metrics for DLBCL (max-pooling).

    Aggregation (protocol.md §2):
        patient_prob = max(patch_probs for same patient_id)
        patient_pred = 1 if patient_prob >= 0.5 else 0

    Returns dict with accuracy, auc, f1, confusion_matrix.
    """
    # Aggregate by patient_id
    patient_probs: dict = defaultdict(list)
    patient_labels: dict = {}

    for prob, label, pid in zip(probs, labels, pids):
        patient_probs[pid].append(prob)
        patient_labels[pid] = label  # all patches share same label

    agg_probs = []
    agg_labels = []
    for pid in patient_probs:
        agg_probs.append(max(patient_probs[pid]))
        agg_labels.append(patient_labels[pid])

    preds = [1 if p >= 0.5 else 0 for p in agg_probs]

    accuracy = sum(p == l for p, l in zip(preds, agg_labels)) / len(agg_labels)
    auc = roc_auc_score(agg_labels, agg_probs)
    f1 = f1_score(agg_labels, preds, average="weighted")
    cm = confusion_matrix(agg_labels, preds).tolist()

    return {
        "accuracy": round(accuracy, 3),
        "auc": round(auc, 3),
        "f1": round(f1, 3),
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(config_path: str, checkpoint_path: str) -> dict:
    """Full evaluation pipeline.

    Returns the evaluation result dict (also written to results/).
    """
    # 1. Load config
    cfg = load_config(config_path)

    # 2. Load checkpoint model_state (validates model compatibility)
    model_state = load_checkpoint(checkpoint_path, cfg)

    # 3. Build model and load weights
    from src.models.classifier import build_classifier

    num_classes = cfg.get("num_classes", 2)
    model = build_classifier(num_classes=num_classes, pretrained=False)

    try:
        model.load_state_dict(model_state)
    except RuntimeError as exc:
        _log.error("Failed to load model_state into model: %s", exc)
        sys.exit(4)

    # 4. Build dataset
    dataset = build_dataset(cfg)

    if len(dataset) == 0:
        _log.error("Dataset is empty — no samples to evaluate")
        sys.exit(2)

    # 5. Run inference
    batch_size = cfg.get("batch_size", 32)
    probs, labels, pids = run_inference(model, dataset, batch_size=batch_size)

    # 6. Compute metrics
    dataset_name = cfg["dataset"]
    if dataset_name == "dlbcl":
        metrics = compute_patient_level_metrics(probs, labels, pids)
    else:
        metrics = compute_image_level_metrics(probs, labels)

    # 7. Build result dict
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "dataset": dataset_name,
        "checkpoint": checkpoint_path,
        "timestamp": timestamp,
        "metrics": metrics,
    }

    # 8. Write output JSON
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", f"{dataset_name}_eval.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    _log.info("Evaluation complete. Results written to: %s", output_path)
    _log.info("Metrics: %s", metrics)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file.")
    args = parser.parse_args()

    evaluate(config_path=args.config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # DoD verification: run full pipeline with fake checkpoint + fake dataset
    # -----------------------------------------------------------------------
    import csv
    import re
    import tempfile
    from collections import OrderedDict

    import numpy as np
    from PIL import Image

    logging.basicConfig(level=logging.INFO)

    print("=== evaluate.py DoD verification ===")

    # ── 1. Create temporary fake images + CSV ────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="eval_smoke_")
    image_paths = []
    for i in range(4):
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img_path = os.path.join(tmp_dir, f"fake_{i}.png")
        Image.fromarray(img_array).save(img_path)
        image_paths.append(img_path)

    csv_path = os.path.join(tmp_dir, "breakhis_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "patient_id", "magnification"])
        writer.writerow([image_paths[0], 0, "SOB_B_A-001", "40x"])
        writer.writerow([image_paths[1], 0, "SOB_B_A-002", "40x"])
        writer.writerow([image_paths[2], 1, "SOB_M_DC-001", "40x"])
        writer.writerow([image_paths[3], 1, "SOB_M_DC-002", "40x"])

    # ── 2. Build fake model and save checkpoint ──────────────────────────
    from src.models.classifier import build_classifier as _build

    fake_model = _build(num_classes=2, pretrained=False)
    ckpt_path = os.path.join(tmp_dir, "fake_best.pth")
    torch.save(
        {
            "epoch": 1,
            "model_state": fake_model.state_dict(),
            "optim_state": OrderedDict(),
            "best_auc": 0.5,
            "config": {"model": "efficientnet_b2"},
        },
        ckpt_path,
    )

    # ── 3. Build fake YAML config ────────────────────────────────────────
    fake_cfg = {
        "dataset": "breakhis",
        "data_root": tmp_dir,
        "split_dir": tmp_dir,
        "magnification": "40x",
        "model": "efficientnet_b2",
        "pretrained": False,
        "num_classes": 2,
        "batch_size": 4,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 1,
        "amp": False,
        "wandb": {"project": "test", "entity": "test"},
    }
    cfg_path = os.path.join(tmp_dir, "fake_breakhis.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(fake_cfg, f)

    # ── 4. Run evaluate ──────────────────────────────────────────────────
    result = evaluate(config_path=cfg_path, checkpoint_path=ckpt_path)

    # ── 5. DoD ①: no errors raised, results file exists ─────────────────
    output_json = os.path.join("results", "breakhis_eval.json")
    assert os.path.exists(output_json), f"Output JSON not found: {output_json}"
    print(f"DoD ①  output file exists: {output_json}")

    # ── 6. DoD ②: JSON contains required keys ────────────────────────────
    with open(output_json) as f:
        loaded = json.load(f)

    assert "accuracy" in loaded["metrics"], "Missing 'accuracy'"
    assert "auc" in loaded["metrics"], "Missing 'auc'"
    assert "f1" in loaded["metrics"], "Missing 'f1'"
    assert "confusion_matrix" in loaded["metrics"], "Missing 'confusion_matrix'"
    print(f"DoD ②  all required metric keys present: {list(loaded['metrics'].keys())}")

    # ── 7. DoD ③: floats are rounded to 3 decimal places ─────────────────
    for key in ("accuracy", "auc", "f1"):
        val = loaded["metrics"][key]
        assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"
        assert round(val, 3) == val, f"{key}={val} is not rounded to 3 decimals"
    print(f"DoD ③  float values rounded to 3 decimals: accuracy={loaded['metrics']['accuracy']}, auc={loaded['metrics']['auc']}, f1={loaded['metrics']['f1']}")

    # ── 8. DoD ④: ISO 8601 UTC timestamp ─────────────────────────────────
    ts = loaded["timestamp"]
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    assert re.match(pattern, ts), f"Timestamp '{ts}' does not match ISO 8601 UTC format"
    print(f"DoD ④  timestamp format valid: {ts}")

    # ── 9. Cleanup ───────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nAll DoD checks passed.")
