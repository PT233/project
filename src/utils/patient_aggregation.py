"""Patient-level aggregation utilities for patch predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


SUPPORTED_AGGREGATIONS = ("max", "mean", "topk_mean", "percentile")


def _maybe_round(value: float, digits: int | None) -> float:
    value = float(value)
    return round(value, digits) if digits is not None else value


def validate_aggregation_method(method: str) -> str:
    """Validate and normalize a patient aggregation method name."""
    normalized = str(method).strip().lower()
    if normalized not in SUPPORTED_AGGREGATIONS:
        supported = ", ".join(SUPPORTED_AGGREGATIONS)
        raise ValueError(f"Unsupported patient aggregation '{method}'. Use one of: {supported}")
    return normalized


def aggregate_patch_probs(
    patch_probs: list[float],
    method: str,
    top_k: int = 10,
    percentile: float = 95.0,
) -> float:
    """Aggregate one patient's patch probabilities into one patient score."""
    method = validate_aggregation_method(method)
    values = np.asarray(patch_probs, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty patch probability list")

    if method == "max":
        return float(values.max())
    if method == "mean":
        return float(values.mean())
    if method == "topk_mean":
        k = max(1, min(int(top_k), values.size))
        return float(np.sort(values)[-k:].mean())
    return float(np.percentile(values, float(percentile)))


def binary_metrics(
    labels: list[int],
    probs: list[float],
    round_digits: int | None = None,
) -> dict[str, Any]:
    """Compute binary metrics with a safe AUC fallback."""
    labels = [int(label) for label in labels]
    probs = [float(prob) for prob in probs]
    preds = [1 if prob >= 0.5 else 0 for prob in probs]

    accuracy = sum(pred == label for pred, label in zip(preds, labels)) / max(len(labels), 1)
    auc = 0.0 if len(set(labels)) < 2 else float(roc_auc_score(labels, probs))
    f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    return {
        "accuracy": _maybe_round(accuracy, round_digits),
        "auc": _maybe_round(auc, round_digits),
        "f1": _maybe_round(f1, round_digits),
        "confusion_matrix": cm,
    }


def compute_patient_aggregation_report(
    probs: list[float],
    labels: list[int],
    patient_ids: list[str],
    selected_method: str = "max",
    top_k: int = 10,
    percentile: float = 95.0,
    round_digits: int | None = None,
    include_patients: bool = True,
) -> dict[str, Any]:
    """Compute patient metrics for max/mean/top-k mean/percentile aggregation."""
    selected_method = validate_aggregation_method(selected_method)
    patient_probs: dict[str, list[float]] = defaultdict(list)
    patient_labels: dict[str, int] = {}
    inconsistent_patients: set[str] = set()

    for prob, label, patient_id in zip(probs, labels, patient_ids):
        pid = str(patient_id)
        label_int = int(label)
        patient_probs[pid].append(float(prob))
        if pid in patient_labels and patient_labels[pid] != label_int:
            inconsistent_patients.add(pid)
        else:
            patient_labels[pid] = label_int

    patient_rows = []
    method_probs: dict[str, list[float]] = {method: [] for method in SUPPORTED_AGGREGATIONS}
    patient_level_labels = []

    for pid in sorted(patient_probs):
        patch_probs = patient_probs[pid]
        row = {
            "patient_id": pid,
            "label": patient_labels[pid],
            "num_patches": len(patch_probs),
        }
        for method in SUPPORTED_AGGREGATIONS:
            value = aggregate_patch_probs(
                patch_probs,
                method=method,
                top_k=top_k,
                percentile=percentile,
            )
            row[method] = _maybe_round(value, round_digits)
            method_probs[method].append(value)

        selected_prob = row[selected_method]
        row["selected_prob"] = selected_prob
        row["selected_pred"] = 1 if float(selected_prob) >= 0.5 else 0
        patient_rows.append(row)
        patient_level_labels.append(patient_labels[pid])

    metrics_by_method = {
        method: binary_metrics(patient_level_labels, values, round_digits=round_digits)
        for method, values in method_probs.items()
    }

    report: dict[str, Any] = {
        "selected": selected_method,
        "top_k": int(top_k),
        "percentile": float(percentile),
        "num_patients": len(patient_rows),
        "inconsistent_patients": sorted(inconsistent_patients),
        "metrics_by_method": metrics_by_method,
    }
    if include_patients:
        report["patients"] = patient_rows
    return report
