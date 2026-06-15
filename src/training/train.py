"""
train.py — Generic training loop with AMP support.

Usage:
    python src/training/train.py --config configs/breakhis.yaml [--resume artifacts/results/checkpoints/last.pth]

Exit codes:
    0 — success
    1 — config error (missing file or invalid fields)
    2 — data path does not exist
    3 — CUDA OOM
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

# Ensure the project root (two levels up from this file) is on sys.path so
# that 'src.*' imports work regardless of how the script is invoked.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.utils.patient_aggregation import compute_patient_aggregation_report

# Logging setup — format: [LEVEL][module] message
_log = logging.getLogger("train")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
_log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False


# Config loading

REQUIRED_FIELDS = [
    "dataset", "data_root", "split_dir", "model", "pretrained",
    "num_classes", "batch_size", "lr", "weight_decay", "epochs", "amp",
]


def load_config(config_path: str) -> dict:
    """Load and validate a YAML config file.

    Returns the config dict. Exits with code 1 on any config error.
    """
    if not os.path.exists(config_path):
        _log.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        _log.error("Failed to parse YAML config: %s", exc)
        sys.exit(1)

    if config is None:
        _log.error("Config file is empty: %s", config_path)
        sys.exit(1)

    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        _log.error("Config missing required fields: %s", missing)
        sys.exit(1)

    if config["dataset"] not in ("breakhis", "dlbcl"):
        _log.error(
            "Invalid dataset value '%s'. Must be 'breakhis' or 'dlbcl'.",
            config["dataset"],
        )
        sys.exit(1)

    return config


# Dataset factory

def _split_filename(config: dict, split: str, default_name: str) -> str:
    split_files = config.get("split_files", {})
    if split_files is None:
        split_files = {}
    if not isinstance(split_files, dict):
        _log.error("split_files must be a mapping, got: %r", split_files)
        sys.exit(1)
    return str(split_files.get(split, default_name))


def build_datasets(config: dict):
    """Instantiate train and val datasets based on config['dataset'].

    Returns (train_dataset, val_dataset). Exits with code 2 if paths are missing.
    """
    dataset_name = config["dataset"]
    data_root = config["data_root"]
    split_dir = config["split_dir"]

    # Validate paths
    if data_root and not os.path.exists(data_root):
        _log.error("data_root does not exist: %s", data_root)
        sys.exit(2)
    if not os.path.exists(split_dir):
        _log.error("split_dir does not exist: %s", split_dir)
        sys.exit(2)

    if dataset_name == "breakhis":
        train_csv = os.path.join(split_dir, _split_filename(config, "train", "train.csv"))
        val_csv = os.path.join(split_dir, _split_filename(config, "val", "val.csv"))
        if not os.path.exists(train_csv):
            _log.error("Train CSV not found: %s", train_csv)
            sys.exit(2)
        if not os.path.exists(val_csv):
            _log.error("Val CSV not found: %s", val_csv)
            sys.exit(2)

        aug = bool(config.get("augmentation", True))
        from src.datasets.breakhis_dataset import BreaKHisDataset
        train_ds = BreaKHisDataset(csv_path=train_csv, data_root=data_root, mode="train",
                                   augmentation=aug)
        val_ds = BreaKHisDataset(csv_path=val_csv, data_root=data_root, mode="val")

    elif dataset_name == "dlbcl":
        train_csv = os.path.join(
            split_dir,
            _split_filename(config, "train", "dlbcl_train.csv"),
        )
        val_csv = os.path.join(
            split_dir,
            _split_filename(config, "val", "dlbcl_val.csv"),
        )
        if not os.path.exists(train_csv):
            _log.error("Train CSV not found: %s", train_csv)
            sys.exit(2)
        if not os.path.exists(val_csv):
            _log.error("Val CSV not found: %s", val_csv)
            sys.exit(2)

        from src.datasets.dlbcl_dataset import DLBCLDataset
        stain_normalization = bool(config.get("stain_normalization", False))
        stain_reference_path = str(config.get("stain_reference_path", "data/stain_reference.png"))
        train_ds = DLBCLDataset(
            csv_path=train_csv,
            data_root=data_root,
            mode="train",
            stain_normalization=stain_normalization,
            stain_reference_path=stain_reference_path,
        )
        val_ds = DLBCLDataset(
            csv_path=val_csv,
            data_root=data_root,
            mode="val",
            stain_normalization=stain_normalization,
            stain_reference_path=stain_reference_path,
        )

    else:
        # Already validated above, but guard anyway
        _log.error("Unknown dataset: %s", dataset_name)
        sys.exit(1)

    return train_ds, val_ds


def _build_patient_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Weight patches inversely by their patient's patch count."""
    patient_counts: dict[str, int] = {}
    patient_ids = [str(pid) for pid in dataset.df["patient_id"].tolist()]
    for pid in patient_ids:
        patient_counts[pid] = patient_counts.get(pid, 0) + 1

    weights = torch.as_tensor(
        [1.0 / patient_counts[pid] for pid in patient_ids],
        dtype=torch.double,
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# One-epoch helpers

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    scaler,  # torch.cuda.amp.GradScaler or None
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """Run one training epoch.

    Returns
    -------
    float
        Mean cross-entropy loss over all batches.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def _safe_auc(labels: list[int], probs: list[float], scope: str) -> float:
    """Compute AUC with a deterministic fallback for single-class inputs."""
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        _log.warning(
            "%s contains only one class (%s); AUC set to 0.0",
            scope,
            unique_labels,
        )
        return 0.0
    return float(roc_auc_score(labels, probs))


def _patient_level_auc(
    labels: list[int],
    probs: list[float],
    patient_ids: list[str],
    aggregation: str = "max",
    top_k: int = 10,
    percentile: float = 95.0,
) -> float:
    """Aggregate patch probabilities by patient, then return selected AUC."""
    report = compute_patient_aggregation_report(
        probs=probs,
        labels=labels,
        patient_ids=patient_ids,
        selected_method=aggregation,
        top_k=top_k,
        percentile=percentile,
        round_digits=None,
        include_patients=False,
    )
    return float(report["metrics_by_method"][report["selected"]]["auc"])


def _patient_level_max_auc(
    labels: list[int],
    probs: list[float],
    patient_ids: list[str],
) -> float:
    """Backward-compatible max-pooling patient AUC helper."""
    return _patient_level_auc(labels, probs, patient_ids, aggregation="max")


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    dataset_name: str,
    patient_aggregation: str = "max",
    patient_top_k: int = 10,
    patient_percentile: float = 95.0,
) -> tuple:
    """Run validation.

    Returns
    -------
    tuple[float, float, float, float | None, dict | None]
        (mean_val_loss, selection_auc, patch_auc, patient_auc, patient_report)
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_labels = []
    all_probs = []
    all_patient_ids = []

    for batch in tqdm(loader, desc="val  ", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if amp_enabled:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(malignant / low-survival)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        if dataset_name == "dlbcl":
            all_patient_ids.extend(str(pid) for pid in batch["patient_id"])

    mean_loss = running_loss / max(n_batches, 1)

    patch_auc = _safe_auc(all_labels, all_probs, "Patch-level validation set")
    patient_auc = None
    patient_report = None
    if dataset_name == "dlbcl":
        patient_report = compute_patient_aggregation_report(
            probs=all_probs,
            labels=all_labels,
            patient_ids=all_patient_ids,
            selected_method=patient_aggregation,
            top_k=patient_top_k,
            percentile=patient_percentile,
            round_digits=None,
            include_patients=False,
        )
        patient_auc = float(patient_report["metrics_by_method"][patient_aggregation]["auc"])
    selection_auc = patient_auc if patient_auc is not None else patch_auc

    return mean_loss, selection_auc, patch_auc, patient_auc, patient_report


# Checkpoint helpers

CKPT_DIR = os.environ.get("CANCER_HISTO_CKPT_DIR", "artifacts/results/checkpoints")


def save_checkpoint(state: dict, path: str) -> None:
    """Save checkpoint dict to path (creates parent dirs if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: AdamW) -> dict:
    """Load model and optimizer states from a checkpoint.

    Returns the full checkpoint dict so callers can read epoch / best_auc.
    Exits with code 1 if the file does not exist.
    """
    if not os.path.exists(path):
        _log.error("Checkpoint not found: %s", path)
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    _log.info(
        "Resumed from %s (epoch %d, best_auc=%.4f)",
        path,
        ckpt.get("epoch", -1),
        ckpt.get("best_auc", 0.0),
    )
    return ckpt


# W&B logger (credential failures are fatal)

def _init_wandb_logger(config: dict):
    """Initialise WandbLogger; return None only for non-exit failures."""
    try:
        from src.utils.logger import WandbLogger
        wandb_cfg = config.get("wandb", {})
        project = wandb_cfg.get("project", f"cancer-histo-{config['dataset']}")
        logger = WandbLogger(config=config, project=project)
        return logger
    except Exception as exc:  # noqa: BLE001
        _log.warning("W&B init failed: %s; continuing without logging.", exc)
        return None


def _log_wandb(logger, metrics: dict) -> None:
    """Log metrics to W&B if logger is available; silently skip on failure."""
    if logger is None:
        return
    try:
        logger.log(metrics)
    except Exception as exc:  # noqa: BLE001
        _log.warning("W&B log failed: %s", exc)


def _finish_wandb(logger) -> None:
    """Finish W&B run if logger is available."""
    if logger is None:
        return
    try:
        logger.finish()
    except Exception as exc:  # noqa: BLE001
        _log.warning("W&B finish failed: %s", exc)


# Main training entry point

def train(config_path: str, resume_path: str | None = None) -> None:
    """Full training loop.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    resume_path : str or None
        Optional path to a checkpoint to resume from.
    """
    config = load_config(config_path)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("Using device: %s", device)

    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    _log.info("AMP enabled: %s", amp_enabled)

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(config)
    _log.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))

    num_workers = int(config.get("num_workers", min(4, os.cpu_count() or 1)))
    patient_balanced_sampling = (
        config["dataset"] == "dlbcl"
        and bool(config.get("patient_balanced_sampling", False))
    )
    train_sampler = _build_patient_balanced_sampler(train_ds) if patient_balanced_sampling else None
    if patient_balanced_sampling:
        _log.info("DLBCL patient-balanced sampler enabled")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    from src.models.classifier import build_classifier
    model = build_classifier(
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
    )
    model = model.to(device)

    # ── Optimizer & scheduler ───────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Loss ────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )

    # ── AMP scaler ──────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auc = 0.0

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, model, optimizer)
        start_epoch = int(ckpt.get("epoch", 0))
        best_auc = float(ckpt.get("best_auc", 0.0))
        # Fast-forward scheduler
        for _ in range(start_epoch):
            scheduler.step()

    # ── W&B ─────────────────────────────────────────────────────────────────
    wb_logger = _init_wandb_logger(config)

    # ── Training loop ────────────────────────────────────────────────────────
    dataset_name = config["dataset"]
    patient_aggregation = str(config.get("patient_aggregation", "max")).strip().lower()
    patient_top_k = int(config.get("patient_top_k", 10))
    patient_percentile = float(config.get("patient_percentile", 95.0))
    early_stop_patience = int(config.get("early_stop_patience", 0))
    epochs_without_improvement = 0
    if dataset_name == "dlbcl":
        _log.info(
            "DLBCL patient aggregation: method=%s top_k=%d percentile=%.1f",
            patient_aggregation,
            patient_top_k,
            patient_percentile,
        )
    if early_stop_patience > 0:
        _log.info("Early stopping enabled: patience=%d", early_stop_patience)
    # DLBCL uses best_dlbcl.pth per task.md #013 spec; BreaKHis uses best.pth
    # config["checkpoint_name"] overrides the default if present
    if config.get("checkpoint_name"):
        best_ckpt_name = config["checkpoint_name"]
    else:
        best_ckpt_name = "best_dlbcl.pth" if dataset_name == "dlbcl" else "best.pth"
    best_ckpt_path = os.path.join(CKPT_DIR, best_ckpt_name)
    last_ckpt_path = os.path.join(CKPT_DIR, "last.pth")

    try:
        for epoch in range(start_epoch, epochs):
            _log.info("Epoch %d / %d", epoch + 1, epochs)

            try:
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, scaler, device, amp_enabled
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _log.error(
                        "CUDA OOM during training. Try reducing batch_size (current=%d).",
                        config["batch_size"],
                    )
                    sys.exit(3)
                raise

            scheduler.step()

            try:
                val_loss, val_auc, val_patch_auc, val_patient_auc, val_patient_report = validate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    amp_enabled,
                    dataset_name,
                    patient_aggregation=patient_aggregation,
                    patient_top_k=patient_top_k,
                    patient_percentile=patient_percentile,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _log.error(
                        "CUDA OOM during validation. Try reducing batch_size (current=%d).",
                        config["batch_size"],
                    )
                    sys.exit(3)
                raise

            if val_patient_auc is None:
                _log.info(
                    "  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  (best_auc=%.4f)",
                    train_loss, val_loss, val_auc, best_auc,
                )
            else:
                patient_metrics = val_patient_report["metrics_by_method"]
                _log.info(
                    "  train_loss=%.4f  val_loss=%.4f  "
                    "val_patch_auc=%.4f  val_patient_%s_auc=%.4f  "
                    "patient_auc[max=%.4f mean=%.4f topk=%.4f pctl=%.4f]  (best_auc=%.4f)",
                    train_loss,
                    val_loss,
                    val_patch_auc,
                    patient_aggregation,
                    val_patient_auc,
                    patient_metrics["max"]["auc"],
                    patient_metrics["mean"]["auc"],
                    patient_metrics["topk_mean"]["auc"],
                    patient_metrics["percentile"]["auc"],
                    best_auc,
                )

            # ── W&B logging ─────────────────────────────────────────────────
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_patch_auc": val_patch_auc,
            }
            if val_patient_auc is not None:
                metrics["val_patient_auc"] = val_patient_auc
                for method, method_metrics in val_patient_report["metrics_by_method"].items():
                    metrics[f"val_patient_{method}_auc"] = method_metrics["auc"]
            _log_wandb(wb_logger, metrics)

            # ── Update best AUC before saving checkpoints ────────────────────
            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # ── Checkpoint state ────────────────────────────────────────────
            ckpt_state: dict = {
                "epoch": epoch + 1,
                "model_state": OrderedDict(model.state_dict()),
                "optim_state": OrderedDict(optimizer.state_dict()),
                "best_auc": best_auc,   # always reflects the running best
                "selection_metric": "patient_auc" if val_patient_auc is not None else "patch_auc",
                "patient_aggregation": patient_aggregation if val_patient_auc is not None else None,
                "patient_top_k": patient_top_k if val_patient_auc is not None else None,
                "patient_percentile": patient_percentile if val_patient_auc is not None else None,
                "val_patient_metrics_by_method": (
                    val_patient_report["metrics_by_method"] if val_patient_report is not None else None
                ),
                "val_patch_auc": val_patch_auc,
                "val_patient_auc": val_patient_auc,
                "config": config,
            }

            # Save last checkpoint every epoch
            save_checkpoint(ckpt_state, last_ckpt_path)

            # Save best checkpoint when val AUC improves
            if is_best:
                save_checkpoint(ckpt_state, best_ckpt_path)
                _log.info(
                    "  New best %s=%.4f — saved %s",
                    ckpt_state["selection_metric"],
                    best_auc,
                    best_ckpt_path,
                )
            elif (
                early_stop_patience > 0
                and epochs_without_improvement >= early_stop_patience
            ):
                _log.info(
                    "Early stopping at epoch %d: no val AUC improvement for %d epoch(s)",
                    epoch + 1,
                    epochs_without_improvement,
                )
                break

    finally:
        _finish_wandb(wb_logger)

    _log.info("Training complete. Best val AUC: %.4f", best_auc)


# CLI entry point

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a classifier on BreaKHis or DLBCL histology data."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="Optional checkpoint path to resume training from.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(config_path=args.config, resume_path=args.resume)
