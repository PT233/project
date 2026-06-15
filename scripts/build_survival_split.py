#!/usr/bin/env python3
"""Build patient-level DLBCL 5-year OS survival splits.

This script intentionally scans the real patch directories under
``data/dlbcl/DLBCL Data/images`` and does not reuse historical DLBCL split CSVs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LOG = logging.getLogger("build_survival_split")
RANDOM_SEED = 42
MIN_PATCHES = 12000
MIN_RATIO = 0.25
MAX_RATIO = 0.75


def _patient_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_survival_labels(clinical_csv: Path) -> pd.DataFrame:
    if not clinical_csv.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")

    clinical = pd.read_csv(clinical_csv)
    required = {"patient_id", "OS", "Follow-up Status"}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"Clinical CSV missing required column(s): {sorted(missing)}")

    labels = clinical[["patient_id", "OS", "Follow-up Status"]].copy()
    labels["patient_id"] = labels["patient_id"].map(_patient_id)
    labels["OS"] = pd.to_numeric(labels["OS"], errors="coerce")
    labels["Follow-up Status"] = pd.to_numeric(
        labels["Follow-up Status"], errors="coerce"
    )
    labels = labels.dropna(subset=["patient_id", "OS", "Follow-up Status"])
    labels["Follow-up Status"] = labels["Follow-up Status"].astype(int)

    early_event = (labels["Follow-up Status"] == 1) & (labels["OS"] < 5)
    five_year_survival = labels["OS"] >= 5
    usable = early_event | five_year_survival
    dropped_censored = int(((labels["Follow-up Status"] == 0) & (labels["OS"] < 5)).sum())
    if dropped_censored:
        LOG.warning(
            "Dropping %d censored patient(s) with OS < 5 years", dropped_censored
        )

    labels = labels.loc[usable].copy()
    labels["label"] = early_event.loc[usable].astype(int)
    labels = labels[["patient_id", "label"]].drop_duplicates("patient_id")
    if labels["label"].nunique() != 2:
        raise ValueError("5-year OS labels must contain both classes")
    return labels


def scan_patches(labels: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing_dirs = 0
    empty_dirs = 0

    for record in labels.sort_values("patient_id").itertuples(index=False):
        patient_dir = image_root / str(record.patient_id)
        if not patient_dir.is_dir():
            missing_dirs += 1
            LOG.warning("Skipping patient %s: patch folder not found", record.patient_id)
            continue

        files = sorted(
            name
            for name in os.listdir(patient_dir)
            if (patient_dir / name).is_file() and name.lower().endswith(".png")
        )
        if not files:
            empty_dirs += 1
            LOG.warning("Skipping patient %s: no PNG patches found", record.patient_id)
            continue

        for name in files:
            rows.append(
                {
                    "filepath": str((patient_dir / name).resolve()),
                    "label": int(record.label),
                    "patient_id": str(record.patient_id),
                }
            )

    if missing_dirs:
        LOG.warning("Skipped %d labeled patient(s) without patch folders", missing_dirs)
    if empty_dirs:
        LOG.warning("Skipped %d labeled patient(s) with empty patch folders", empty_dirs)
    if not rows:
        raise ValueError(f"No PNG patches found under {image_root}")

    patches = pd.DataFrame(rows).sort_values(["patient_id", "filepath"]).reset_index(drop=True)
    patient_counts = patches.drop_duplicates("patient_id")["label"].value_counts().to_dict()
    LOG.info("Scanned %d patches from %d labeled patients", len(patches), patches["patient_id"].nunique())
    LOG.info("Available patient label distribution: %s", patient_counts)
    return patches


def select_balanced_patients(
    patches: pd.DataFrame,
    random_seed: int,
    majority_to_minority: int = 2,
) -> pd.DataFrame:
    """Keep all minority-class patients and a reproducible subset of majority patients.

    The 5-year OS labels are highly imbalanced in this cohort. Selecting a 2:1
    majority/minority patient set keeps all high-risk survival patients while
    making each patient-level holdout satisfy the requested 25%-75% label range.
    """

    patients = (
        patches[["patient_id", "label"]]
        .drop_duplicates("patient_id")
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    label_counts = patients["label"].value_counts()
    if label_counts.min() < 2:
        raise ValueError(f"Not enough patients per label for stratified split: {label_counts.to_dict()}")

    minority_label = int(label_counts.idxmin())
    majority_label = int(label_counts.idxmax())
    minority = patients[patients["label"] == minority_label]
    majority = patients[patients["label"] == majority_label]
    majority_target = min(len(majority), majority_to_minority * len(minority))
    majority = majority.sample(n=majority_target, random_state=random_seed)

    selected = (
        pd.concat([minority, majority], ignore_index=True)
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    LOG.info(
        "Selected %d patients for splitting; label distribution=%s",
        len(selected),
        selected["label"].value_counts().sort_index().to_dict(),
    )
    return selected


def stratified_patient_split(
    patients: pd.DataFrame,
    random_seed: int,
) -> dict[str, set[str]]:
    train_df, temp_df = train_test_split(
        patients,
        test_size=0.30,
        stratify=patients["label"],
        random_state=random_seed,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=random_seed,
        shuffle=True,
    )
    return {
        "train": set(train_df["patient_id"]),
        "val": set(val_df["patient_id"]),
        "test": set(test_df["patient_id"]),
    }


def validate_and_write(
    patches: pd.DataFrame,
    split_map: dict[str, set[str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total_patches = 0
    split_frames: dict[str, pd.DataFrame] = {}

    for split_name in ("train", "val", "test"):
        patient_ids = split_map[split_name]
        overlap = seen & patient_ids
        if overlap:
            raise AssertionError(
                f"Patient leakage detected in {split_name}: {sorted(overlap)[:10]}"
            )
        seen.update(patient_ids)

        split_df = patches[patches["patient_id"].isin(patient_ids)].copy()
        if split_df.empty:
            raise ValueError(f"{split_name} split is empty")

        patient_labels = split_df.drop_duplicates("patient_id")["label"].value_counts()
        patch_labels = split_df["label"].value_counts()
        for scope, counts in (("patient", patient_labels), ("patch", patch_labels)):
            ratio = float(counts.min() / counts.sum())
            if ratio < MIN_RATIO or ratio > MAX_RATIO:
                raise AssertionError(
                    f"{split_name} {scope} label ratio out of range: {counts.to_dict()}"
                )

        split_df = split_df[["filepath", "label", "patient_id"]].sort_values(
            ["patient_id", "filepath"]
        )
        split_frames[split_name] = split_df
        total_patches += len(split_df)

    if total_patches < MIN_PATCHES:
        raise AssertionError(
            f"Total patch count {total_patches} is below required minimum {MIN_PATCHES}"
        )

    for split_name, split_df in split_frames.items():
        path = output_dir / f"dlbcl_survival_{split_name}.csv"
        split_df.to_csv(path, index=False)
        patient_labels = (
            split_df.drop_duplicates("patient_id")["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        patch_labels = split_df["label"].value_counts().sort_index().to_dict()
        print(
            f"{split_name}: patients={split_df['patient_id'].nunique()}, "
            f"patches={len(split_df)}, "
            f"patient_label_distribution={patient_labels}, "
            f"patch_label_distribution={patch_labels}"
        )

    print(f"total_patches={total_patches}")


def build(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    clinical_csv = Path(args.clinical_csv)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    if not clinical_csv.is_absolute():
        clinical_csv = project_root / clinical_csv
    if not image_root.is_absolute():
        image_root = project_root / image_root
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    labels = load_survival_labels(clinical_csv)
    patches = scan_patches(labels, image_root)
    selected_patients = select_balanced_patients(
        patches,
        random_seed=int(args.random_seed),
        majority_to_minority=int(args.majority_to_minority),
    )
    selected_patches = patches[patches["patient_id"].isin(selected_patients["patient_id"])]
    split_map = stratified_patient_split(selected_patients, random_seed=int(args.random_seed))
    validate_and_write(selected_patches, split_map, output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--clinical-csv",
        default="data/dlbcl/clinical_data_cleaned.csv",
    )
    parser.add_argument(
        "--image-root",
        default="data/dlbcl/DLBCL Data/images",
    )
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--majority-to-minority",
        type=int,
        default=2,
        help="Patient-level majority/minority ratio after deterministic downsampling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        build(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Failed to build DLBCL survival splits: %s", exc)
        print("FAILED")
        return 1

    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
