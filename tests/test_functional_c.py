"""
test_functional_c.py — Functional tests for datasets / train / evaluate
FT-01 to FT-12 as specified in test.md §2.1–§2.4
All tests use fake/tempfile data — no real dataset required.
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON = sys.executable  # python inside conda env

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_fake_image(path: str, size=(224, 224)):
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def make_breakhis_csv(csv_path: str, image_paths: list, magnification="40x"):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "patient_id", "magnification"])
        for i, img_path in enumerate(image_paths):
            label = i % 2
            pid = f"SOB_P_{i:02d}"
            writer.writerow([img_path, label, pid, magnification])


def make_dlbcl_csv(csv_path: str, image_paths: list):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "patient_id", "magnification"])
        patients = ["patient_001", "patient_001", "patient_002", "patient_002"]
        labels   = [1, 1, 0, 0]
        for i, img_path in enumerate(image_paths):
            pid = patients[i % len(patients)]
            lbl = labels[i % len(labels)]
            writer.writerow([img_path, lbl, pid, "N/A"])


def make_train_val_csvs(split_dir: str, image_paths: list):
    """Create train.csv and val.csv in split_dir for breakhis training."""
    # train: first N-2 images, val: last 2 images (ensure both classes present)
    # We need at least 4 images for train (drop_last=True requires batch >= 1)
    train_paths = image_paths[:-2]
    val_paths = image_paths[-2:]

    for split_name, paths in [("train", train_paths), ("val", val_paths)]:
        csv_path = os.path.join(split_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "patient_id", "magnification"])
            for i, img_path in enumerate(paths):
                label = i % 2
                pid = f"SOB_P_{split_name}_{i:02d}"
                writer.writerow([img_path, label, pid, "40x"])


def make_fake_config(config_path: str, data_root: str, split_dir: str,
                     epochs: int = 1, batch_size: int = 2) -> dict:
    cfg = {
        "dataset": "breakhis",
        "data_root": data_root,
        "split_dir": split_dir,
        "magnification": "40x",
        "model": "efficientnet_b2",
        "pretrained": False,
        "num_classes": 2,
        "batch_size": batch_size,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": epochs,
        "amp": False,
        "augmentation": False,
        "wandb": {"project": "test", "entity": "test"},
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg


# ---------------------------------------------------------------------------
# Shared fixture: fake training environment (reused by FT-06 to FT-12)
# ---------------------------------------------------------------------------

# We use a module-scoped fixture so the expensive train run happens once.
_TRAIN_ENV = {}  # populated by ft06_result fixture


@pytest.fixture(scope="module")
def fake_train_env():
    """Create a persistent temp dir with fake images + CSVs for training tests."""
    tmpdir = tempfile.mkdtemp(prefix="ft_train_")
    data_root = os.path.join(tmpdir, "images")
    split_dir = os.path.join(tmpdir, "splits")
    ckpt_dir = os.path.join(tmpdir, "checkpoints")
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Generate 8 fake images (4 train + 2 val with both classes)
    n_images = 8
    image_paths = []
    for i in range(n_images):
        img_path = os.path.join(data_root, f"fake_{i:02d}.png")
        make_fake_image(img_path)
        image_paths.append(img_path)

    make_train_val_csvs(split_dir, image_paths)

    config_path = os.path.join(tmpdir, "config.yaml")
    make_fake_config(
        config_path,
        data_root=data_root,
        split_dir=split_dir,
        epochs=1,
        batch_size=2,
    )

    yield {
        "tmpdir": tmpdir,
        "data_root": data_root,
        "split_dir": split_dir,
        "ckpt_dir": ckpt_dir,
        "config_path": config_path,
        "image_paths": image_paths,
    }

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def ft06_result(fake_train_env):
    """Run train.py once (epochs=1) and cache result for FT-06/07/09/10/11/12."""
    env = fake_train_env.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    env["WANDB_MODE"] = "offline"
    env["WANDB_API_KEY"] = "fake_key_for_testing"
    env["PATH"] = os.environ.get("PATH", "")
    env["HOME"] = os.environ.get("HOME", "")
    env["CONDA_DEFAULT_ENV"] = os.environ.get("CONDA_DEFAULT_ENV", "")
    env["CONDA_PREFIX"] = os.environ.get("CONDA_PREFIX", "")

    train_script = os.path.join(PROJECT_ROOT, "src", "training", "train.py")
    config_path = fake_train_env["config_path"]
    ckpt_dir = fake_train_env["ckpt_dir"]

    last_pth = os.path.join(ckpt_dir, "last.pth")

    # Patch CKPT_DIR by passing env var — train.py saves to results/checkpoints
    # We override by monkeypatching via a wrapper script or env var trick.
    # Since train.py uses module-level CKPT_DIR = "results/checkpoints",
    # we need to run from the tmpdir so the relative path lands in tmpdir.
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, train_script, "--config", config_path],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "WANDB_MODE": "offline", "WANDB_API_KEY": "fake_key",
             "PYTHONPATH": PROJECT_ROOT},
        cwd=fake_train_env["tmpdir"],
    )
    elapsed = time.time() - t0

    # Determine actual last.pth path: train.py uses CKPT_DIR="results/checkpoints"
    # relative to CWD which is fake_train_env["tmpdir"]
    last_pth_actual = os.path.join(fake_train_env["tmpdir"], "results", "checkpoints", "last.pth")

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed": elapsed,
        "last_pth": last_pth_actual,
        "tmpdir": fake_train_env["tmpdir"],
        "config_path": config_path,
    }


# ===========================================================================
# FT-01: breakhis_dataset.py __main__ prints shape: torch.Size([3, 224, 224])
# ===========================================================================

def test_ft01_breakhis_stdout_shape():
    """FT-01: Running breakhis_dataset.py prints expected shape."""
    script = os.path.join(PROJECT_ROOT, "src", "datasets", "breakhis_dataset.py")
    result = subprocess.run(
        [PYTHON, script],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"breakhis_dataset.py exited with {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    assert "shape: torch.Size([3, 224, 224])" in result.stdout, (
        f"Expected shape output not found in stdout:\n{result.stdout}"
    )


# ===========================================================================
# FT-02: CSV with non-existent filepath → exit code 2
# ===========================================================================

def test_ft02_breakhis_missing_image_exit2():
    """FT-02: Dataset with missing image file exits with code 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "patient_id", "magnification"])
            writer.writerow(["/nonexistent/path/image.png", 0, "SOB_P_00", "40x"])

        # Write a small runner script that iterates the dataset
        runner = os.path.join(tmpdir, "runner.py")
        runner_code = f"""
import sys
sys.path.insert(0, "{PROJECT_ROOT}")
from src.datasets.breakhis_dataset import BreaKHisDataset

ds = BreaKHisDataset(csv_path="{csv_path}", data_root="{tmpdir}", mode="val")
# Trigger __getitem__ to hit sys.exit(2)
_ = ds[0]
"""
        with open(runner, "w") as f:
            f.write(runner_code)

        result = subprocess.run(
            [PYTHON, runner],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        assert result.returncode == 2, (
            f"Expected exit code 2, got {result.returncode}\n"
            f"stderr: {result.stderr}"
        )


# ===========================================================================
# FT-03: DataLoader batch has non-empty patient_id strings
# ===========================================================================

def test_ft03_breakhis_dataloader_patient_id():
    """FT-03: DataLoader batch contains non-empty patient_id strings."""
    from src.datasets.breakhis_dataset import BreaKHisDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths = []
        for i in range(4):
            p = os.path.join(tmpdir, f"img_{i}.png")
            make_fake_image(p)
            img_paths.append(p)

        csv_path = os.path.join(tmpdir, "split.csv")
        make_breakhis_csv(csv_path, img_paths, magnification="40x")

        ds = BreaKHisDataset(csv_path=csv_path, data_root=tmpdir, mode="val",
                             augmentation=False)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        batch = next(iter(loader))
        pids = batch["patient_id"]

        assert len(pids) > 0, "patient_id batch is empty"
        for pid in pids:
            assert isinstance(pid, str) and len(pid) > 0, (
                f"patient_id should be non-empty string, got: {repr(pid)}"
            )


# ===========================================================================
# FT-04: dlbcl_dataset.py __main__ prints shape and label
# ===========================================================================

def test_ft04_dlbcl_stdout():
    """FT-04: Running dlbcl_dataset.py prints shape and label."""
    script = os.path.join(PROJECT_ROOT, "src", "datasets", "dlbcl_dataset.py")
    result = subprocess.run(
        [PYTHON, script],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"dlbcl_dataset.py exited with {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    # Expect "Sample shape: torch.Size([3, 224, 224]), label: <int>"
    assert "shape" in result.stdout.lower(), (
        f"Expected 'shape' in stdout:\n{result.stdout}"
    )
    assert "label" in result.stdout.lower(), (
        f"Expected 'label' in stdout:\n{result.stdout}"
    )


# ===========================================================================
# FT-05: DLBCLDataset patient counts and get_patient_ids()
# ===========================================================================

def test_ft05_dlbcl_patient_ids():
    """FT-05: DLBCLDataset get_patient_ids() returns list; each has >= 1 sample."""
    from src.datasets.dlbcl_dataset import DLBCLDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths = []
        for i in range(4):
            p = os.path.join(tmpdir, f"patch_{i:03d}.png")
            make_fake_image(p)
            img_paths.append(p)

        csv_path = os.path.join(tmpdir, "fake_split.csv")
        make_dlbcl_csv(csv_path, img_paths)

        ds = DLBCLDataset(csv_path=csv_path, data_root=tmpdir, mode="val")

        pids = ds.get_patient_ids()
        assert isinstance(pids, list), "get_patient_ids() should return a list"
        assert len(pids) > 0, "get_patient_ids() returned empty list"

        for pid in pids:
            patches = ds.get_patches_by_patient(pid)
            assert len(patches) >= 1, (
                f"Patient {pid} has no patches"
            )


# ===========================================================================
# FT-06: train.py exits 0 with fake data (epochs=1)
# ===========================================================================

def test_ft06_train_exits_zero(ft06_result):
    """FT-06: train.py with fake data (epochs=1) exits with code 0."""
    rc = ft06_result["returncode"]
    assert rc == 0, (
        f"train.py exited with {rc}\n"
        f"stdout: {ft06_result['stdout']}\n"
        f"stderr: {ft06_result['stderr']}"
    )


# ===========================================================================
# FT-07: last.pth exists and is loadable
# ===========================================================================

def test_ft07_checkpoint_exists(ft06_result):
    """FT-07: After training, last.pth exists and torch.load succeeds."""
    last_pth = ft06_result["last_pth"]
    assert os.path.exists(last_pth), (
        f"last.pth not found at {last_pth}\n"
        f"train stderr: {ft06_result['stderr']}"
    )

    ckpt = torch.load(last_pth, map_location="cpu")
    assert isinstance(ckpt, dict), "Checkpoint should be a dict"
    required_keys = {"epoch", "model_state", "optim_state", "best_auc", "config"}
    missing = required_keys - set(ckpt.keys())
    assert not missing, f"Checkpoint missing keys: {missing}"


# ===========================================================================
# FT-08: non-existent config → exit code 1
# ===========================================================================

def test_ft08_missing_config_exit1():
    """FT-08: train.py with non-existent config exits with code 1."""
    train_script = os.path.join(PROJECT_ROOT, "src", "training", "train.py")
    result = subprocess.run(
        [PYTHON, train_script, "--config", "/nonexistent/path/config.yaml"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
    )
    assert result.returncode == 1, (
        f"Expected exit code 1, got {result.returncode}\n"
        f"stderr: {result.stderr}"
    )


# ===========================================================================
# FT-09: --resume continues from saved checkpoint (epoch field >= 1)
# ===========================================================================

def test_ft09_resume_checkpoint(ft06_result, fake_train_env):
    """FT-09: Resuming from last.pth with epochs=2 produces epoch >= 1 in ckpt."""
    last_pth = ft06_result["last_pth"]
    if not os.path.exists(last_pth):
        pytest.skip("last.pth not found (FT-07 may have failed)")

    # Create a config with epochs=2 to ensure the resumed run completes an epoch
    with tempfile.TemporaryDirectory() as tmpdir2:
        config_path2 = os.path.join(tmpdir2, "config2.yaml")
        make_fake_config(
            config_path2,
            data_root=fake_train_env["data_root"],
            split_dir=fake_train_env["split_dir"],
            epochs=2,
            batch_size=2,
        )

        train_script = os.path.join(PROJECT_ROOT, "src", "training", "train.py")
        result = subprocess.run(
            [PYTHON, train_script, "--config", config_path2, "--resume", last_pth],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "WANDB_MODE": "offline", "WANDB_API_KEY": "fake_key",
                 "PYTHONPATH": PROJECT_ROOT},
            cwd=tmpdir2,
        )
        assert result.returncode == 0, (
            f"Resumed train.py exited with {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

        new_last_pth = os.path.join(tmpdir2, "results", "checkpoints", "last.pth")
        assert os.path.exists(new_last_pth), (
            f"new last.pth not found at {new_last_pth}\n"
            f"stderr: {result.stderr}"
        )

        ckpt = torch.load(new_last_pth, map_location="cpu")
        epoch_val = ckpt.get("epoch", 0)
        assert epoch_val >= 1, (
            f"Expected epoch >= 1 in resumed checkpoint, got {epoch_val}"
        )


# ===========================================================================
# FT-10: evaluate.py exits 0, JSON output exists
# ===========================================================================

@pytest.fixture(scope="module")
def ft10_result(ft06_result, fake_train_env):
    """Run evaluate.py once and cache result for FT-10/11/12."""
    last_pth = ft06_result["last_pth"]
    if not os.path.exists(last_pth):
        return {"skipped": True, "reason": "last.pth not found"}

    # Build eval config (uses val.csv as test split fallback)
    eval_tmpdir = tempfile.mkdtemp(prefix="ft_eval_")
    config_path = os.path.join(eval_tmpdir, "eval_config.yaml")

    cfg = {
        "dataset": "breakhis",
        "data_root": fake_train_env["data_root"],
        "split_dir": fake_train_env["split_dir"],
        "magnification": "40x",
        "model": "efficientnet_b2",
        "pretrained": False,
        "num_classes": 2,
        "batch_size": 2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 1,
        "amp": False,
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    eval_script = os.path.join(PROJECT_ROOT, "src", "training", "evaluate.py")
    result = subprocess.run(
        [PYTHON, eval_script, "--config", config_path, "--checkpoint", last_pth],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        cwd=eval_tmpdir,
    )

    # JSON is written to results/breakhis_eval.json relative to cwd
    json_path = os.path.join(eval_tmpdir, "results", "breakhis_eval.json")

    return {
        "skipped": False,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "json_path": json_path,
        "eval_tmpdir": eval_tmpdir,
    }


def test_ft10_evaluate_exits_zero(ft10_result):
    """FT-10: evaluate.py exits 0 and JSON output file exists."""
    if ft10_result.get("skipped"):
        pytest.skip(ft10_result.get("reason", "FT-10 skipped"))

    rc = ft10_result["returncode"]
    assert rc == 0, (
        f"evaluate.py exited with {rc}\n"
        f"stdout: {ft10_result['stdout']}\n"
        f"stderr: {ft10_result['stderr']}"
    )

    json_path = ft10_result["json_path"]
    assert os.path.exists(json_path), (
        f"Expected JSON output at {json_path}\n"
        f"stderr: {ft10_result['stderr']}"
    )


# ===========================================================================
# FT-11: JSON contains required metrics with 3-decimal-place floats
# ===========================================================================

def test_ft11_json_metrics(ft10_result):
    """FT-11: JSON has accuracy/auc/f1/confusion_matrix; floats rounded to 3 dp."""
    if ft10_result.get("skipped"):
        pytest.skip(ft10_result.get("reason", "FT-11 skipped"))

    json_path = ft10_result["json_path"]
    if not os.path.exists(json_path):
        pytest.skip("JSON file not found (FT-10 may have failed)")

    with open(json_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    assert "accuracy" in metrics, "metrics missing 'accuracy'"
    assert "auc" in metrics, "metrics missing 'auc'"
    assert "f1" in metrics, "metrics missing 'f1'"
    assert "confusion_matrix" in metrics, "metrics missing 'confusion_matrix'"

    for key in ("accuracy", "auc", "f1"):
        val = metrics[key]
        assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"
        assert round(val, 3) == val, (
            f"{key}={val} is not rounded to 3 decimal places"
        )


# ===========================================================================
# FT-12: JSON timestamp matches ISO 8601 format
# ===========================================================================

def test_ft12_json_timestamp(ft10_result):
    """FT-12: JSON timestamp field matches ^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$."""
    if ft10_result.get("skipped"):
        pytest.skip(ft10_result.get("reason", "FT-12 skipped"))

    json_path = ft10_result["json_path"]
    if not os.path.exists(json_path):
        pytest.skip("JSON file not found (FT-10 may have failed)")

    with open(json_path) as f:
        data = json.load(f)

    ts = data.get("timestamp", "")
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    assert re.match(pattern, ts), (
        f"Timestamp '{ts}' does not match ISO 8601 UTC format "
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    )
