"""
train.py — Generic training loop with AMP support.

Usage:
    python src/training/train.py --config configs/breakhis.yaml [--resume results/checkpoints/last.pth]

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
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup — format: [LEVEL][module] message
# ---------------------------------------------------------------------------
_log = logging.getLogger("train")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
_log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

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
        train_csv = os.path.join(split_dir, "train.csv")
        val_csv = os.path.join(split_dir, "val.csv")
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
        train_csv = os.path.join(split_dir, "dlbcl_train.csv")
        val_csv = os.path.join(split_dir, "dlbcl_val.csv")
        if not os.path.exists(train_csv):
            _log.error("Train CSV not found: %s", train_csv)
            sys.exit(2)
        if not os.path.exists(val_csv):
            _log.error("Val CSV not found: %s", val_csv)
            sys.exit(2)

        from src.datasets.dlbcl_dataset import DLBCLDataset
        train_ds = DLBCLDataset(csv_path=train_csv, data_root=data_root, mode="train")
        val_ds = DLBCLDataset(csv_path=val_csv, data_root=data_root, mode="val")

    else:
        # Already validated above, but guard anyway
        _log.error("Unknown dataset: %s", dataset_name)
        sys.exit(1)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

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


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple:
    """Run validation.

    Returns
    -------
    tuple[float, float]
        (mean_val_loss, val_auc)
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_labels = []
    all_probs = []

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

    mean_loss = running_loss / max(n_batches, 1)

    # Compute AUC; guard against edge cases (single class in val set)
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        _log.warning(
            "Validation set contains only one class (%s); AUC set to 0.0",
            unique_labels,
        )
        auc = 0.0
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return mean_loss, auc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CKPT_DIR = "results/checkpoints"


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


# ---------------------------------------------------------------------------
# W&B logger (optional — failures are non-fatal)
# ---------------------------------------------------------------------------

def _init_wandb_logger(config: dict):
    """Try to initialise WandbLogger; return None on failure."""
    try:
        from src.utils.logger import WandbLogger
        wandb_cfg = config.get("wandb", {})
        project = wandb_cfg.get("project", f"cancer-histo-{config['dataset']}")
        logger = WandbLogger(config=config, project=project)
        return logger
    except SystemExit:
        _log.warning("W&B init failed (likely missing WANDB_API_KEY); continuing without logging.")
        return None
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


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

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

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
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
    criterion = nn.CrossEntropyLoss()

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
    # DLBCL uses best_dlbcl.pth per task.md #013 DoD; BreaKHis uses best.pth
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
                val_loss, val_auc = validate(model, val_loader, criterion, device, amp_enabled)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _log.error(
                        "CUDA OOM during validation. Try reducing batch_size (current=%d).",
                        config["batch_size"],
                    )
                    sys.exit(3)
                raise

            _log.info(
                "  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  (best_auc=%.4f)",
                train_loss, val_loss, val_auc, best_auc,
            )

            # ── W&B logging ─────────────────────────────────────────────────
            _log_wandb(wb_logger, {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            })

            # ── Update best AUC before saving checkpoints ────────────────────
            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc

            # ── Checkpoint state ────────────────────────────────────────────
            ckpt_state: dict = {
                "epoch": epoch + 1,
                "model_state": OrderedDict(model.state_dict()),
                "optim_state": OrderedDict(optimizer.state_dict()),
                "best_auc": best_auc,   # always reflects the running best
                "config": config,
            }

            # Save last checkpoint every epoch
            save_checkpoint(ckpt_state, last_ckpt_path)

            # Save best checkpoint when val AUC improves
            if is_best:
                save_checkpoint(ckpt_state, best_ckpt_path)
                _log.info("  New best AUC=%.4f — saved %s", best_auc, best_ckpt_path)

    finally:
        _finish_wandb(wb_logger)

    _log.info("Training complete. Best val AUC: %.4f", best_auc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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


def _dod_selftest() -> None:
    """DoD __main__ verification.

    Runs a single epoch with fake data (batch_size=4, epochs=1) to confirm
    the training loop works end-to-end and last.pth is saved correctly.
    This function mutates the module-level CKPT_DIR temporarily.
    """
    import csv
    import shutil
    import tempfile

    import numpy as np
    from PIL import Image

    global CKPT_DIR  # noqa: PLW0603

    logging.basicConfig(level=logging.INFO)
    print("=== DoD __main__ verification ===")

    # ── Build a minimal temporary environment ──────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="train_dod_")
    data_root = os.path.join(tmpdir, "images")
    split_dir = os.path.join(tmpdir, "splits")
    ckpt_dir = os.path.join(tmpdir, "checkpoints")
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Generate 10 fake PNG images (5 benign + 5 malignant)
    rows = []
    for i in range(10):
        label = i % 2
        fname = f"fake_{i:02d}.png"
        fpath = os.path.join(data_root, fname)
        img = Image.fromarray(
            np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        )
        img.save(fpath)
        rows.append(
            {
                "filepath": fpath,
                "label": label,
                "patient_id": f"SOB_P{i:02d}",
                "magnification": "40x",
            }
        )

    # Write train and val CSV splits
    fieldnames = ["filepath", "label", "patient_id", "magnification"]
    for split_name, split_rows in [("train", rows[:8]), ("val", rows[8:])]:
        csv_path = os.path.join(split_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    # Minimal config (epochs=1, batch_size=4, no AMP, no W&B)
    dod_config = {
        "dataset": "breakhis",
        "data_root": data_root,
        "split_dir": split_dir,
        "magnification": "40x",
        "model": "efficientnet_b2",
        "pretrained": False,       # avoid download during test
        "num_classes": 2,
        "batch_size": 4,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 1,
        "amp": False,
        "wandb": {"project": "test", "entity": "test"},
    }
    config_path = os.path.join(tmpdir, "dod_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(dod_config, f)

    # Temporarily redirect checkpoint save location to tmpdir
    original_ckpt_dir = CKPT_DIR
    CKPT_DIR = ckpt_dir

    try:
        train(config_path=config_path, resume_path=None)

        # ── DoD ③: verify last.pth exists and has required fields ──────────
        last_pth = os.path.join(ckpt_dir, "last.pth")
        assert os.path.exists(last_pth), f"last.pth not found at {last_pth}"

        ckpt = torch.load(last_pth, map_location="cpu")
        required_keys = {"epoch", "model_state", "optim_state", "best_auc", "config"}
        missing_keys = required_keys - set(ckpt.keys())
        assert not missing_keys, f"last.pth missing keys: {missing_keys}"
        assert isinstance(ckpt["epoch"], int), "epoch must be int"
        assert isinstance(ckpt["best_auc"], float), "best_auc must be float"
        assert isinstance(ckpt["config"], dict), "config must be dict"

        print(f"DoD ③  last.pth exists and has keys: {set(ckpt.keys())}")
        print(f"       epoch={ckpt['epoch']}, best_auc={ckpt['best_auc']:.4f}")
        print("=== All DoD checks passed ===")

    finally:
        CKPT_DIR = original_ckpt_dir
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    # If --config is provided, behave as the normal CLI training launcher.
    # If run with no arguments (or only unrecognised flags), run the DoD
    # self-test using temporary fake data.
    if "--config" in sys.argv:
        args = _parse_args()
        train(config_path=args.config, resume_path=args.resume)
    else:
        _dod_selftest()
