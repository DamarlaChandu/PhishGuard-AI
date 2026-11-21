"""Tests for independent model evaluation on synthetic external inputs."""

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.features import FEATURE_NAMES  # noqa: E402
from scripts.evaluate_external_models import (  # noqa: E402
    BASELINE_METRICS_PATH,
    EXTERNAL_DATASET_PATH,
    MODEL_METADATA_PATH,
    MODEL_NAMES,
    MODELS_DIR,
    OUTPUT_DIR,
    compute_external_metrics,
    run_evaluation,
)


class FixedProbabilityModel:
    classes_ = np.array([0, 1])
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self, feature_index: int = 0) -> None:
        self.feature_index = feature_index

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        phishing_scores = X[:, self.feature_index] / 100
        return np.column_stack([1 - phishing_scores, phishing_scores])


class InvalidProbabilityModel:
    classes_ = np.array([0, 1])
    n_features_in_ = len(FEATURE_NAMES)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 1), 0.5)


def write_synthetic_project(root: Path) -> None:
    (root / MODELS_DIR).mkdir(parents=True)
    (root / EXTERNAL_DATASET_PATH).parent.mkdir(parents=True)
    (root / BASELINE_METRICS_PATH).parent.mkdir(parents=True)

    with (root / EXTERNAL_DATASET_PATH).open("w", encoding="utf-8", newline="") as handle:
        columns = ["url", "domain", "label", "source", "source_record_id", *FEATURE_NAMES]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        rows = [
            (0, 10, 1, 1, "https://good.example/", "good.example"),
            (1, 80, 0, 6, "http://bad.example/login", "bad.example"),
            (0, 40, 0, 6, "http://shop.example/cart", "shop.example"),
            (1, 90, 1, 1, "https://phish.example/", "phish.example"),
        ]
        for index, (label, score, https, path_length, url, domain) in enumerate(rows):
            row = {
                "url": url,
                "domain": domain,
                "label": label,
                "source": "compphish",
                "source_record_id": f"record-{index}",
                **{name: 0 for name in FEATURE_NAMES},
            }
            row["url_length"] = score
            row["has_https"] = https
            row["path_length"] = path_length
            writer.writerow(row)

    metadata = {
        "label_convention": {"1": "phishing", "0": "legitimate"},
        "positive_class": 1,
        "decision_threshold": 0.5,
        "feature_names": list(FEATURE_NAMES),
        "models": {
            name: (MODELS_DIR / f"{name}.joblib").as_posix() for name in MODEL_NAMES
        },
    }
    (root / MODEL_METADATA_PATH).write_text(json.dumps(metadata), encoding="utf-8")

    baseline_model_metrics = {
        "n_samples": 4,
        "n_phishing": 2,
        "n_legitimate": 2,
        "threshold": 0.5,
        "accuracy": 0.75,
        "precision_phishing": 1.0,
        "recall_phishing": 0.5,
        "f1_phishing": 2 / 3,
        "precision_legitimate": 2 / 3,
        "recall_legitimate": 1.0,
        "f1_legitimate": 0.8,
        "f1_macro": 11 / 15,
        "roc_auc": 0.9,
        "true_negatives": 2,
        "false_positives": 0,
        "false_negatives": 1,
        "true_positives": 1,
    }
    baseline = {
        "dataset_sha256": "synthetic-baseline",
        "test": {name: baseline_model_metrics for name in MODEL_NAMES},
    }
    (root / BASELINE_METRICS_PATH).write_text(json.dumps(baseline), encoding="utf-8")

    for name in MODEL_NAMES:
        joblib.dump(FixedProbabilityModel(), root / MODELS_DIR / f"{name}.joblib")


def test_compute_external_metrics_calculates_rates_and_pr_auc() -> None:
    result = compute_external_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.2, 0.9]),
    )

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["false_negative_rate"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["pr_auc"] == pytest.approx(5 / 6)
    assert result["per_class"]["phishing"]["support"] == 2
    assert result["per_class"]["legitimate"]["support"] == 2


def test_metrics_handle_empty_and_single_class_subgroups() -> None:
    empty = compute_external_metrics(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
    )
    assert empty["support"] == 0
    assert empty["accuracy"] is None
    assert empty["roc_auc"] is None
    assert empty["pr_auc"] is None
    assert empty["confusion_matrix"] == [[0, 0], [0, 0]]

    phishing_only = compute_external_metrics(
        np.array([1, 1]), np.array([1, 0]), np.array([0.9, 0.2])
    )
    assert phishing_only["roc_auc"] is None
    assert phishing_only["pr_auc"] is None
    assert phishing_only["false_positive_rate"] is None
    assert phishing_only["false_negative_rate"] == pytest.approx(0.5)


def test_run_writes_metrics_report_and_prediction_rows(tmp_path: Path) -> None:
    write_synthetic_project(tmp_path)

    result = run_evaluation(tmp_path)

    assert result["evaluation"]["feature_names"] == list(FEATURE_NAMES)
    assert result["evaluation"]["threshold"] == 0.5
    for name in MODEL_NAMES:
        assert result["models"][name]["overall"]["support"] == 4
        assert result["models"][name]["comparison_to_baseline"]["accuracy"]["baseline"] == 0.75
    assert set(result["subgroups"]["https"]) == {"https", "http"}
    assert result["subgroups"]["https"]["http"]["support"] == 2

    output_dir = tmp_path / OUTPUT_DIR
    metrics_path = output_dir / "compphish_external_metrics.json"
    report_path = output_dir / "compphish_external_report.txt"
    predictions_path = output_dir / "compphish_external_predictions.csv"
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics["models"][MODEL_NAMES[0]]["overall"]["confusion_matrix"] == [
        [2, 0],
        [0, 2],
    ]
    report = report_path.read_text(encoding="utf-8")
    assert "labels have not been independently verified" in report
    assert "does not, by itself, establish a model implementation defect" in report
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["url"] == "https://good.example/"
    assert rows[0]["domain"] == "good.example"
    assert rows[0]["true_label"] == "0"
    assert rows[0]["source_record_id"] == "record-0"
    assert rows[0]["random_forest_score"] == "0.1"
    assert rows[0]["random_forest_predicted_label"] == "0"


def test_run_handles_an_empty_subgroup(tmp_path: Path) -> None:
    write_synthetic_project(tmp_path)
    dataset_path = tmp_path / EXTERNAL_DATASET_PATH
    with dataset_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = rows[0].keys()
    for row in rows:
        row["has_https"] = "1"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    result = run_evaluation(tmp_path)

    empty_http = result["subgroups"]["https"]["http"]
    assert empty_http["support"] == 0
    assert empty_http["models"][MODEL_NAMES[0]]["accuracy"] is None
    assert empty_http["models"][MODEL_NAMES[0]]["roc_auc"] is None


def test_missing_model_file_fails_explicitly(tmp_path: Path) -> None:
    write_synthetic_project(tmp_path)
    (tmp_path / MODELS_DIR / "logistic_regression.joblib").unlink()

    with pytest.raises(FileNotFoundError):
        run_evaluation(tmp_path)


def test_invalid_model_probability_shape_fails_explicitly(tmp_path: Path) -> None:
    write_synthetic_project(tmp_path)
    joblib.dump(InvalidProbabilityModel(), tmp_path / MODELS_DIR / "random_forest.joblib")

    with pytest.raises(ValueError, match="probability matrix shape"):
        run_evaluation(tmp_path)


def test_metadata_with_wrong_feature_order_is_rejected(tmp_path: Path) -> None:
    write_synthetic_project(tmp_path)
    metadata_path = tmp_path / MODEL_METADATA_PATH
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_names"] = list(reversed(FEATURE_NAMES))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="FEATURE_NAMES order"):
        run_evaluation(tmp_path)


def test_invalid_external_feature_value_fails_explicitly(tmp_path: Path) -> None:
    write_synthetic_project(tmp_path)
    dataset_path = tmp_path / EXTERNAL_DATASET_PATH
    with dataset_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = rows[0].keys()
    rows[0]["num_dots"] = "not-an-integer"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="num_dots must be a non-negative integer"):
        run_evaluation(tmp_path)


def test_invalid_binary_feature_value_fails_explicitly(tmp_path: Path) -> None:
    write_synthetic_project(tmp_path)
    dataset_path = tmp_path / EXTERNAL_DATASET_PATH
    with dataset_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = rows[0].keys()
    rows[0]["has_https"] = "2"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="has_https must be 0 or 1"):
        run_evaluation(tmp_path)
