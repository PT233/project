"""
evaluate.py — Model evaluation script.

Loads a trained checkpoint and evaluates it on the test split.

Usage:
    python src/training/evaluate.py --config configs/breakhis.yaml --checkpoint artifacts/results/checkpoints/best.pth

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
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so 'src.*' imports work regardless of cwd.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import yaml
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from src.utils.patient_aggregation import compute_patient_aggregation_report

# Module-level logger — format: [ERROR][evaluate] <message>
_log = logging.getLogger("evaluate")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
_log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False

RESULTS_DIR = os.environ.get("CANCER_HISTO_RESULTS_DIR", "artifacts/results")


# Helper: load YAML config

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


# Helper: load checkpoint

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


# Helper: build dataset

def _split_filename(cfg: dict, split: str, default_name: str) -> str:
    split_files = cfg.get("split_files", {})
    if split_files is None:
        split_files = {}
    if not isinstance(split_files, dict):
        _log.error("split_files must be a mapping, got: %r", split_files)
        sys.exit(1)
    return str(split_files.get(split, default_name))


def build_dataset(cfg: dict, split: str = "test"):
    """Instantiate the correct dataset class for the val/test split.

    Exits with code 2 if data paths do not exist.
    Returns the dataset instance.
    """
    dataset_name = cfg["dataset"]
    split_dir = cfg["split_dir"]
    data_root = cfg["data_root"]
    split = str(split).strip().lower()
    if split not in {"train", "val", "test"}:
        _log.error("Invalid split '%s'. Expected train, val, or test.", split)
        sys.exit(1)

    if not os.path.exists(split_dir):
        _log.error("split_dir does not exist: %s", split_dir)
        sys.exit(2)

    if data_root and not os.path.exists(data_root):
        _log.error("data_root does not exist: %s", data_root)
        sys.exit(2)

    dataset_defaults = {
        "breakhis": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "breakhis_test.csv",
        },
        "dlbcl": {
            "train": "dlbcl_train.csv",
            "val": "dlbcl_val.csv",
            "test": "dlbcl_test.csv",
        },
    }
    split_files = cfg.get("split_files", {}) or {}
    has_explicit_split = isinstance(split_files, dict) and split in split_files
    csv_candidates = [
        os.path.join(
            split_dir,
            _split_filename(cfg, split, dataset_defaults[dataset_name][split]),
        )
    ]
    if has_explicit_split:
        pass
    elif split == "test":
        csv_candidates.extend([
            os.path.join(split_dir, f"{dataset_name}_test.csv"),
            os.path.join(split_dir, f"{dataset_name}_val.csv"),
            os.path.join(split_dir, "test.csv"),
            os.path.join(split_dir, "val.csv"),
        ])
    elif split == "val":
        csv_candidates.extend([
            os.path.join(split_dir, f"{dataset_name}_val.csv"),
            os.path.join(split_dir, "val.csv"),
        ])
    else:
        csv_candidates.extend([
            os.path.join(split_dir, f"{dataset_name}_train.csv"),
            os.path.join(split_dir, "train.csv"),
        ])
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
        dataset = DLBCLDataset(
            csv_path=csv_path,
            data_root=data_root,
            mode="val",
            stain_normalization=bool(cfg.get("stain_normalization", False)),
            stain_reference_path=str(cfg.get("stain_reference_path", "data/stain_reference.png")),
        )

    return dataset


# Helper: run inference

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


# Helper: compute metrics

def _safe_roc_auc(labels, probs, metric_scope: str) -> float:
    """Return 0.0 when AUC is undefined for a single-class evaluation set."""
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        _log.warning(
            "%s evaluation set contains only one class (%s); AUC set to 0.0",
            metric_scope,
            unique_labels,
        )
        return 0.0

    return roc_auc_score(labels, probs)


def compute_image_level_metrics(probs, labels):
    """Compute image-level evaluation metrics.

    Returns dict with accuracy, auc, f1, confusion_matrix.
    """
    preds = [1 if p >= 0.5 else 0 for p in probs]

    accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    auc = _safe_roc_auc(labels, probs, "Image-level")
    f1 = f1_score(labels, preds, average="weighted")
    cm = confusion_matrix(labels, preds).tolist()

    return {
        "accuracy": round(accuracy, 3),
        "auc": round(auc, 3),
        "f1": round(f1, 3),
        "confusion_matrix": cm,
    }


def compute_patient_level_metrics(
    probs,
    labels,
    pids,
    aggregation: str = "max",
    top_k: int = 10,
    percentile: float = 95.0,
    return_report: bool = False,
):
    """Compute patient-level evaluation metrics for DLBCL.

    The returned ``metrics`` uses the selected aggregation method. The
    companion report includes max/mean/top-k mean/percentile diagnostics.

    Returns metrics by default, or (metrics, aggregation_report) when requested.
    """
    report = compute_patient_aggregation_report(
        probs=probs,
        labels=labels,
        patient_ids=pids,
        selected_method=aggregation,
        top_k=top_k,
        percentile=percentile,
        round_digits=3,
        include_patients=True,
    )
    metrics = report["metrics_by_method"][report["selected"]]
    if return_report:
        return metrics, report
    return metrics


# Main evaluation function

def evaluate(config_path: str, checkpoint_path: str, split: str = "test") -> dict:
    """Full evaluation pipeline.

    Returns the evaluation result dict (also written to artifacts/results/).
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
    dataset = build_dataset(cfg, split=split)

    if len(dataset) == 0:
        _log.error("Dataset is empty — no samples to evaluate")
        sys.exit(2)

    # 5. Run inference
    batch_size = cfg.get("batch_size", 32)
    probs, labels, pids = run_inference(model, dataset, batch_size=batch_size)

    # 6. Compute metrics
    dataset_name = cfg["dataset"]
    if dataset_name == "dlbcl":
        aggregation = cfg.get("patient_aggregation", "max")
        top_k = int(cfg.get("patient_top_k", 10))
        percentile = float(cfg.get("patient_percentile", 95.0))
        metrics, patient_aggregation = compute_patient_level_metrics(
            probs,
            labels,
            pids,
            aggregation=aggregation,
            top_k=top_k,
            percentile=percentile,
            return_report=True,
        )
    else:
        metrics = compute_image_level_metrics(probs, labels)
        patient_aggregation = None

    # 7. Build result dict
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "dataset": dataset_name,
        "split": split,
        "split_csv": getattr(dataset, "csv_path", None),
        "checkpoint": checkpoint_path,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    if patient_aggregation is not None:
        result["patient_aggregation"] = patient_aggregation

    # 8. Write output JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{split}"
    output_path = os.path.join(RESULTS_DIR, f"{dataset_name}_eval{suffix}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    if split == "test":
        legacy_output_path = os.path.join(RESULTS_DIR, f"{dataset_name}_eval.json")
        with open(legacy_output_path, "w") as f:
            json.dump(result, f, indent=2)

    _log.info("Evaluation complete. Results written to: %s", output_path)
    _log.info("Metrics: %s", metrics)

    return result


# CLI entry point

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate. Defaults to test, with legacy val fallback if no test CSV exists.",
    )
    args = parser.parse_args()

    evaluate(config_path=args.config, checkpoint_path=args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
