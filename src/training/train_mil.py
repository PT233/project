"""train_mil.py — end-to-end ABMIL training for DLBCL (bag-level).

Independent of train.py. Trains EfficientNet-B2 + gated attention MIL on
patient bags, selecting the best checkpoint by patient-level (= bag-level) AUC.

Usage:
    python src/training/train_mil.py --config configs/dlbcl_mil.yaml
"""

import argparse
import logging
import os
import sys

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from src.datasets.dlbcl_bag_dataset import DLBCLBagDataset
from src.models.mil import build_mil_model

logging.basicConfig(level=logging.INFO, format="[%(levelname)s][mil] %(message)s")
_log = logging.getLogger("mil")

CKPT_DIR = os.environ.get("CANCER_HISTO_CKPT_DIR", "artifacts/results/checkpoints")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(cfg, split_file, mode, bag_size):
    ds = DLBCLBagDataset(
        csv_path=os.path.join(cfg["split_dir"], split_file),
        data_root=cfg["data_root"],
        mode=mode,
        bag_size=bag_size,
        stain_normalization=bool(cfg.get("stain_normalization", False)),
        stain_reference_path=cfg.get("stain_reference_path", "data/stain_reference.png"),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg.get("bag_batch_size", 2)),
        shuffle=(mode == "train"),
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(model, loader, criterion, device, amp, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    torch.set_grad_enabled(train)
    desc = "train" if train else "eval "
    pbar = tqdm(loader, desc=desc, ascii=True, leave=False)
    for batch in pbar:
        bags = batch["bag"].to(device)          # (B, N, 3, 224, 224)
        labels = batch["label"].to(device)      # (B,)
        if train:
            optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=amp):
            logits, _ = model(bags)
            loss = criterion(logits, labels)
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += float(loss) * bags.size(0)
        n += bags.size(0)
        pbar.set_postfix(loss=f"{total_loss / max(n, 1):.4f}")
        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    torch.set_grad_enabled(True)
    return total_loss / max(n, 1), np.array(all_labels), np.array(all_probs)


def metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels).mean())
    auc = 0.0 if len(set(labels.tolist())) < 2 else float(roc_auc_score(labels, probs))
    f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {"accuracy": round(acc, 4), "auc": round(auc, 4),
            "f1": round(f1, 4), "confusion_matrix": cm}


def main(config_path):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    bag_size = int(cfg.get("bag_size", 64))
    _log.info("device=%s amp=%s bag_size=%d", device, amp, bag_size)

    train_loader = make_loader(cfg, cfg["split_files"]["train"], "train", bag_size)
    val_loader = make_loader(cfg, cfg["split_files"]["val"], "val", bag_size)
    _log.info("train_bags=%d val_bags=%d", len(train_loader.dataset), len(val_loader.dataset))

    model = build_mil_model(
        num_classes=int(cfg.get("num_classes", 2)),
        pretrained=bool(cfg.get("pretrained", True)),
        attn_hidden=int(cfg.get("attn_hidden", 128)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]),
                      weight_decay=float(cfg["weight_decay"]))
    epochs = int(cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    scaler = torch.amp.GradScaler("cuda") if amp else None

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_path = os.path.join(CKPT_DIR, cfg.get("checkpoint_name", "best_dlbcl_mil.pth"))
    best_auc = 0.0
    patience = int(cfg.get("early_stop_patience", 0))
    no_improve = 0

    for epoch in range(epochs):
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, device, amp,
                                  optimizer=optimizer, scaler=scaler)
        scheduler.step()
        va_loss, va_labels, va_probs = run_epoch(model, val_loader, criterion, device, amp)
        m = metrics(va_labels, va_probs)
        _log.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_patient_auc=%.4f  acc=%.4f  f1=%.4f",
                  epoch + 1, epochs, tr_loss, va_loss, m["auc"], m["accuracy"], m["f1"])
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            no_improve = 0
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "best_auc": best_auc, "config": cfg}, best_path)
            _log.info("  New best val_patient_auc=%.4f -> saved %s", best_auc, best_path)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                _log.info("Early stopping at epoch %d (no improve %d)", epoch + 1, no_improve)
                break

    _log.info("Training complete. Best val_patient_auc=%.4f", best_auc)

    # Evaluate best checkpoint on the TEST split and append results to markdown.
    if cfg["split_files"].get("test"):
        _log.info("Evaluating best checkpoint on test split...")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        test_loader = make_loader(cfg, cfg["split_files"]["test"], "val", bag_size)
        _, te_labels, te_probs = run_epoch(model, test_loader, criterion, device, amp)
        tm = metrics(te_labels, te_probs)
        _log.info("TEST patient_auc=%.4f acc=%.4f f1=%.4f cm=%s",
                  tm["auc"], tm["accuracy"], tm["f1"], tm["confusion_matrix"])
        import datetime, json
        md_path = os.path.join(cfg["data_root"], "DLBCL_MIL_RESULT.md")
        lines = []
        lines.append("")
        lines.append("# DLBCL ABMIL result " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append("| model | EfficientNet-B2 + gated ABMIL (end-to-end) |")
        lines.append("| bag_size | " + str(bag_size) + " |")
        lines.append("| best_val_patient_auc | " + str(best_auc) + " |")
        lines.append("| test_accuracy | " + str(tm["accuracy"]) + " |")
        lines.append("| test_patient_auc | " + str(tm["auc"]) + " |")
        lines.append("| test_f1 | " + str(tm["f1"]) + " |")
        lines.append("| test_confusion_matrix | " + json.dumps(tm["confusion_matrix"]) + " |")
        lines.append("| meets_target(auc>=0.75) | " + ("yes" if tm["auc"] >= 0.75 else "no") + " |")
        lines.append("")
        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        _log.info("Wrote results to %s", md_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
