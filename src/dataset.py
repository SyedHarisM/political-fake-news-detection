"""LIAR dataset acquisition, label-safe conversion, and statistics."""

from __future__ import annotations

import json
import logging
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.paths import PROCESSED_DIR, RESULTS_DIR, ensure_project_directories

LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
HF_DATASET_IDS = ("ucsbnlp/liar", "liar")
ORIGINAL_ZIP_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"mostly-true", "true"}
EXCLUDED_LABELS = {"half-true"}
BINARY_LABEL_NAMES = {0: "FAKE", 1: "REAL"}

TSV_COLUMNS = [
    "id",
    "original_label",
    "statement",
    "subject",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]


class DatasetAcquisitionError(RuntimeError):
    """Raised when no supported LIAR source can be acquired."""


def _normalize_label_name(label: Any) -> str:
    return str(label).strip().lower().replace("_", "-").replace("pants-fire!", "pants-fire")


def _load_from_huggingface() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load LIAR and derive label names from its declared feature metadata."""
    from datasets import load_dataset

    errors: list[str] = []
    for dataset_id in HF_DATASET_IDS:
        try:
            LOGGER.info("Attempting Hugging Face dataset: %s", dataset_id)
            dataset = load_dataset(dataset_id)
            splits: dict[str, pd.DataFrame] = {}
            label_names: list[str] | None = None
            schema: dict[str, str] = {}
            for split_name, split in dataset.items():
                if split_name not in {"train", "validation", "test"}:
                    continue
                feature = split.features.get("label")
                if feature is not None and getattr(feature, "names", None):
                    label_names = list(feature.names)
                schema = {name: repr(feature_value) for name, feature_value in split.features.items()}
                frame = split.to_pandas()
                if "statement" not in frame.columns or "label" not in frame.columns:
                    raise ValueError(f"Unexpected LIAR schema: {list(frame.columns)}")
                if pd.api.types.is_integer_dtype(frame["label"]):
                    if not label_names:
                        raise ValueError("Integer labels found without ClassLabel names")
                    frame["original_label"] = frame["label"].map(
                        lambda idx: label_names[int(idx)] if pd.notna(idx) else None
                    )
                else:
                    frame["original_label"] = frame["label"].astype(str)
                splits[split_name] = frame
            required = {"train", "validation", "test"}
            if not required.issubset(splits):
                raise ValueError(f"Missing official splits: {required - set(splits)}")
            return splits, {
                "source": f"Hugging Face: {dataset_id}",
                "dataset_id": dataset_id,
                "schema": schema,
                "declared_label_names": label_names,
            }
        except Exception as exc:  # try the next documented source
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
            LOGGER.warning("Hugging Face source failed: %s", errors[-1])
    raise DatasetAcquisitionError("; ".join(errors))


def _load_from_original_zip() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fallback to the dataset authors' original TSV archive."""
    response = requests.get(ORIGINAL_ZIP_URL, timeout=90)
    response.raise_for_status()
    splits: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        for split_name in ("train", "validation", "test"):
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith(f"{split_name}.tsv")
                or (split_name == "validation" and name.lower().endswith("valid.tsv"))
            ]
            if not candidates:
                raise DatasetAcquisitionError(f"Archive lacks the {split_name} TSV split")
            with archive.open(candidates[0]) as handle:
                splits[split_name] = pd.read_csv(handle, sep="\t", header=None, names=TSV_COLUMNS)
    return splits, {
        "source": "LIAR authors' original TSV archive",
        "url": ORIGINAL_ZIP_URL,
        "schema": {column: "TSV column" for column in TSV_COLUMNS},
        "declared_label_names": sorted(FAKE_LABELS | REAL_LABELS | EXCLUDED_LABELS),
    }


def acquire_raw_splits() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Acquire official splits, preferring Hugging Face with an author-host fallback."""
    try:
        return _load_from_huggingface()
    except Exception as hf_error:
        LOGGER.warning("All Hugging Face attempts failed; trying original archive: %s", hf_error)
        try:
            return _load_from_original_zip()
        except Exception as archive_error:
            raise DatasetAcquisitionError(
                "Could not acquire LIAR from Hugging Face or the original archive. "
                f"Hugging Face errors: {hf_error}. Archive error: {archive_error}"
            ) from archive_error


def _count_labels(values: pd.Series) -> dict[str, int]:
    return dict(sorted(Counter(values.dropna().astype(str)).items()))


def prepare_dataset(force_download: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Download, validate, map, save, and summarize the official LIAR splits."""
    ensure_project_directories()
    cached_paths = {name: PROCESSED_DIR / f"{name}.csv" for name in ("train", "validation", "test")}
    stats_path = RESULTS_DIR / "dataset_statistics.json"
    if not force_download and all(path.exists() for path in cached_paths.values()) and stats_path.exists():
        frames = {name: pd.read_csv(path) for name, path in cached_paths.items()}
        return frames, json.loads(stats_path.read_text(encoding="utf-8"))

    raw_splits, source_metadata = acquire_raw_splits()
    processed: dict[str, pd.DataFrame] = {}
    original_distributions: dict[str, dict[str, int]] = {}
    binary_distributions: dict[str, dict[str, int]] = {}
    excluded_counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}

    allowed_labels = FAKE_LABELS | REAL_LABELS | EXCLUDED_LABELS
    for split_name in ("train", "validation", "test"):
        frame = raw_splits[split_name].copy()
        raw_counts[split_name] = len(frame)
        frame["original_label"] = frame["original_label"].map(_normalize_label_name)
        discovered = set(frame["original_label"].dropna().unique())
        unexpected = discovered - allowed_labels
        if unexpected:
            raise ValueError(f"Unexpected LIAR labels in {split_name}: {sorted(unexpected)}")
        original_distributions[split_name] = _count_labels(frame["original_label"])
        excluded_counts[split_name] = int(frame["original_label"].isin(EXCLUDED_LABELS).sum())
        frame = frame[frame["original_label"].isin(FAKE_LABELS | REAL_LABELS)].copy()
        frame = frame[frame["statement"].notna()].copy()
        frame["statement"] = frame["statement"].astype(str).str.strip()
        frame = frame[frame["statement"].str.len() > 0].copy()
        frame["binary_label"] = frame["original_label"].map(
            lambda label: 0 if label in FAKE_LABELS else 1
        ).astype(int)
        keep_first = ["statement", "original_label", "binary_label"]
        other_columns = [column for column in frame.columns if column not in keep_first and column != "label"]
        frame = frame[keep_first + other_columns].reset_index(drop=True)
        binary_distributions[split_name] = {
            BINARY_LABEL_NAMES[int(label)]: int(count)
            for label, count in frame["binary_label"].value_counts().sort_index().items()
        }
        processed[split_name] = frame
        frame.to_csv(cached_paths[split_name], index=False)

    stats: dict[str, Any] = {
        "dataset": "LIAR",
        "source": source_metadata,
        "random_seed": RANDOM_SEED,
        "mapping": {
            "FAKE (0)": sorted(FAKE_LABELS),
            "REAL (1)": sorted(REAL_LABELS),
            "excluded": sorted(EXCLUDED_LABELS),
        },
        "raw_split_counts": raw_counts,
        "processed_split_counts": {name: len(frame) for name, frame in processed.items()},
        "excluded_half_true_counts": excluded_counts,
        "original_class_distributions": original_distributions,
        "binary_class_distributions": binary_distributions,
        "total_raw_samples": sum(raw_counts.values()),
        "total_processed_samples": sum(len(frame) for frame in processed.values()),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    processed["test"].head(12)[["statement", "original_label", "binary_label"]].to_csv(
        PROCESSED_DIR / "held_out_examples.csv", index=False
    )
    return processed, stats

