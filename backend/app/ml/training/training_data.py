"""Load the prepared feature dataset and split it by domain group.

Model inputs are exactly the 15 backend URL features (FEATURE_NAMES order).
Metadata columns are loaded only for grouping and analysis, never as inputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from app.ml.features import FEATURE_NAMES
from app.ml.features.domain import domain_group_key

TARGET_COLUMN = "label"
METADATA_COLUMNS: tuple[str, ...] = ("url", "domain", "source", "collected_at", "cleaning_actions")
REQUIRED_COLUMNS: tuple[str, ...] = (*METADATA_COLUMNS, TARGET_COLUMN, *FEATURE_NAMES)
VALID_LABELS = frozenset({0, 1})


@dataclass(frozen=True)
class FeatureDataset:
    """Feature matrix, target and per-row metadata, aligned by index."""

    X: np.ndarray  # shape (n_rows, len(FEATURE_NAMES)), float64
    y: np.ndarray  # shape (n_rows,), int64; 1 = phishing, 0 = legitimate
    hostnames: np.ndarray  # str, from the "domain" column
    groups: np.ndarray  # str, registrable domain used for splitting
    sources: np.ndarray  # str
    cleaning_actions: np.ndarray  # str, ";"-separated
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, indices: np.ndarray) -> FeatureDataset:
        return FeatureDataset(
            X=self.X[indices],
            y=self.y[indices],
            hostnames=self.hostnames[indices],
            groups=self.groups[indices],
            sources=self.sources[indices],
            cleaning_actions=self.cleaning_actions[indices],
            feature_names=self.feature_names,
        )


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    test: np.ndarray


def registrable_domain(hostname: str) -> str:
    """Return the registrable domain used as a split group.

    Hosts without an eTLD+1 (such as IPs, localhost, and malformed names) use
    their normalized host string as a group key so unrelated values do not
    collapse into one empty group.
    """
    return domain_group_key(hostname)


def load_feature_dataset(path: Path) -> FeatureDataset:
    """Read the prepared CSV and validate columns, labels and feature values.

    Raises:
        ValueError: if a required column is missing, a label is not 0/1, or a
            feature value is not a non-negative integer.
    """
    features: list[list[int]] = []
    labels: list[int] = []
    hostnames: list[str] = []
    sources: list[str] = []
    actions: list[str] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        for row in reader:
            line = reader.line_num
            label = _parse_int(row[TARGET_COLUMN], TARGET_COLUMN, line)
            if label not in VALID_LABELS:
                raise ValueError(f"line {line}: label must be 0 or 1, got {label}")
            values = [_parse_int(row[name], name, line) for name in FEATURE_NAMES]
            if any(value < 0 for value in values):
                raise ValueError(f"line {line}: feature values must be non-negative")
            features.append(values)
            labels.append(label)
            hostnames.append(row["domain"])
            sources.append(row["source"])
            actions.append(row["cleaning_actions"] or "")

    if not labels:
        raise ValueError(f"{path.name} contains no rows")

    hostname_array = np.array(hostnames, dtype=str)
    return FeatureDataset(
        X=np.array(features, dtype=np.float64),
        y=np.array(labels, dtype=np.int64),
        hostnames=hostname_array,
        groups=np.array([registrable_domain(h) for h in hostnames], dtype=str),
        sources=np.array(sources, dtype=str),
        cleaning_actions=np.array(actions, dtype=str),
    )


def _parse_int(value: str, column: str, line: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"line {line}: {column} must be an integer, got {value!r}") from None


def group_stratified_split(
    y: np.ndarray, groups: np.ndarray, test_fraction: float = 0.2, seed: int = 42
) -> SplitIndices:
    """Split so that no group appears in both sets, keeping class ratios close.

    Uses one fold of ``StratifiedGroupKFold`` with ``round(1 / test_fraction)``
    folds, shuffled with ``seed`` for reproducibility.
    """
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    n_splits = round(1 / test_fraction)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"need at least {n_splits} distinct groups to split")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(y)), y, groups))
    return SplitIndices(train=np.sort(train), test=np.sort(test))


def overlapping_values(a: np.ndarray, b: np.ndarray) -> set[str]:
    """Values present in both arrays (used to verify no host/group leakage)."""
    return set(a.tolist()) & set(b.tolist())
