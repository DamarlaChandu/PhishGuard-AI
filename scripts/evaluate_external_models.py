"""Evaluate the saved baseline models on the independent CompPhish dataset.

Run from the project root with the project's ML dependencies available:
    python scripts/evaluate_external_models.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from app.ml.features import FEATURE_NAMES  # noqa: E402
from app.ml.training.baseline_models import LOGISTIC_REGRESSION, RANDOM_FOREST  # noqa: E402
from app.ml.training.evaluation import DEFAULT_THRESHOLD, compute_metrics  # noqa: E402

EXTERNAL_DATASET_PATH = Path("data/processed/compphish_external_test.csv")
MODELS_DIR = Path("models/baseline")
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
BASELINE_METRICS_PATH = Path("reports/training/baseline_metrics.json")
OUTPUT_DIR = Path("reports/evaluation")
MODEL_NAMES = (RANDOM_FOREST, LOGISTIC_REGRESSION)
BINARY_FEATURES = (
    "has_https",
    "has_ip_address",
    "has_suspicious_keywords",
    "has_at_symbol",
    "has_port",
    "is_shortened_url",
)
PREDICTION_COLUMNS = (
    "url",
    "domain",
    "true_label",
    "source",
    "source_record_id",
    "random_forest_score",
    "random_forest_predicted_label",
    "logistic_regression_score",
    "logistic_regression_predicted_label",
)
COMPARISON_METRICS = (
    "accuracy",
    "precision_phishing",
    "recall_phishing",
    "f1_phishing",
    "precision_legitimate",
    "recall_legitimate",
    "f1_legitimate",
    "f1_macro",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
    "false_negative_rate",
)


@dataclass(frozen=True)
class ExternalDataset:
    X: np.ndarray
    y: np.ndarray
    urls: tuple[str, ...]
    domains: tuple[str, ...]
    sources: tuple[str, ...]
    source_record_ids: tuple[str, ...]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_dataset(path: Path) -> ExternalDataset:
    features: list[list[int]] = []
    labels: list[int] = []
    urls: list[str] = []
    domains: list[str] = []
    sources: list[str] = []
    source_record_ids: list[str] = []
    required = {"url", "domain", "label", "source", "source_record_id", *FEATURE_NAMES}

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no CSV header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path.name} has duplicate column names")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        for row in reader:
            line = reader.line_num
            label_text = (row["label"] or "").strip()
            if label_text not in {"0", "1"}:
                raise ValueError(f"line {line}: label must be 0 or 1, got {label_text!r}")
            metadata = {
                name: (row[name] or "").strip()
                for name in ("url", "domain", "source", "source_record_id")
            }
            empty_metadata = [name for name, value in metadata.items() if not value]
            if empty_metadata:
                raise ValueError(f"line {line}: empty required values: {empty_metadata}")
            row_features = []
            for name in FEATURE_NAMES:
                value_text = (row[name] or "").strip()
                try:
                    value = int(value_text)
                except ValueError:
                    raise ValueError(
                        f"line {line}: {name} must be a non-negative integer, got {value_text!r}"
                    ) from None
                if value < 0:
                    raise ValueError(f"line {line}: {name} must be non-negative")
                if name in BINARY_FEATURES and value not in {0, 1}:
                    raise ValueError(f"line {line}: {name} must be 0 or 1")
                row_features.append(value)
            features.append(row_features)
            labels.append(int(label_text))
            urls.append(metadata["url"])
            domains.append(metadata["domain"])
            sources.append(metadata["source"])
            source_record_ids.append(metadata["source_record_id"])

    if not labels:
        raise ValueError(f"{path.name} contains no records")
    return ExternalDataset(
        X=np.asarray(features, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        urls=tuple(urls),
        domains=tuple(domains),
        sources=tuple(sources),
        source_record_ids=tuple(source_record_ids),
    )


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def validate_model_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("model metadata feature_names do not match FEATURE_NAMES order")
    if metadata.get("decision_threshold") != DEFAULT_THRESHOLD:
        raise ValueError("model metadata decision_threshold must be 0.5")
    if metadata.get("positive_class") != 1:
        raise ValueError("model metadata positive_class must be 1 (phishing)")
    if metadata.get("label_convention") != {"1": "phishing", "0": "legitimate"}:
        raise ValueError("model metadata label convention is not supported")
    expected_models = {
        RANDOM_FOREST: (MODELS_DIR / f"{RANDOM_FOREST}.joblib").as_posix(),
        LOGISTIC_REGRESSION: (MODELS_DIR / f"{LOGISTIC_REGRESSION}.joblib").as_posix(),
    }
    if metadata.get("models") != expected_models:
        raise ValueError("model metadata paths do not match the expected baseline models")


def load_models(project_root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name in MODEL_NAMES:
        path = project_root / MODELS_DIR / f"{name}.joblib"
        model = joblib.load(path)
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError(f"{path} does not provide predict_proba")
        classes = getattr(model, "classes_", None)
        if classes is None or len(classes) != 2 or set(classes) != {0, 1}:
            raise ValueError(f"{path} must be fitted with classes 0 and 1")
        feature_count = getattr(model, "n_features_in_", len(FEATURE_NAMES))
        if feature_count != len(FEATURE_NAMES):
            raise ValueError(
                f"{path} expects {feature_count} features; expected {len(FEATURE_NAMES)}"
            )
        models[name] = model
    return models


def validate_baseline_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    test_metrics = metrics.get("test")
    if not isinstance(test_metrics, dict):
        raise ValueError("baseline metrics are missing the test results")
    for name in MODEL_NAMES:
        result = test_metrics.get(name)
        if not isinstance(result, dict):
            raise ValueError(f"baseline metrics are missing test results for {name}")
        if result.get("threshold") != DEFAULT_THRESHOLD:
            raise ValueError(f"baseline metrics for {name} do not use threshold 0.5")
    return test_metrics


def model_phishing_scores(model: Any, X: np.ndarray, model_name: str) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = list(model.classes_)
    if probabilities.shape != (len(X), len(classes)):
        raise ValueError(f"{model_name} returned an invalid probability matrix shape")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError(f"{model_name} returned invalid probability values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{model_name} probabilities do not sum to 1")
    phishing_scores = probabilities[:, classes.index(1)]
    if not np.all(np.isfinite(phishing_scores)):
        raise ValueError(f"{model_name} returned non-finite probability scores")
    if np.any((phishing_scores < 0) | (phishing_scores > 1)):
        raise ValueError(f"{model_name} returned probability scores outside [0, 1]")
    return phishing_scores


def compute_external_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phishing_scores: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    phishing_scores = np.asarray(phishing_scores, dtype=np.float64)
    if y_true.ndim != 1 or y_pred.ndim != 1 or phishing_scores.ndim != 1:
        raise ValueError("labels, predictions and scores must be one-dimensional")
    if len(y_true) != len(y_pred) or len(y_true) != len(phishing_scores):
        raise ValueError("labels, predictions and scores must have matching lengths")
    if np.any(~np.isin(y_true, [0, 1])) or np.any(~np.isin(y_pred, [0, 1])):
        raise ValueError("labels and predictions must contain only 0 or 1")
    if not np.all(np.isfinite(phishing_scores)) or np.any(
        (phishing_scores < 0) | (phishing_scores > 1)
    ):
        raise ValueError("phishing scores must be finite probabilities in [0, 1]")

    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "support": 0,
            "n_phishing": 0,
            "n_legitimate": 0,
            "threshold": DEFAULT_THRESHOLD,
            "accuracy": None,
            "precision_phishing": None,
            "recall_phishing": None,
            "f1_phishing": None,
            "precision_legitimate": None,
            "recall_legitimate": None,
            "f1_legitimate": None,
            "f1_macro": None,
            "roc_auc": None,
            "pr_auc": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "per_class": {
                "legitimate": {
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "support": 0,
                },
                "phishing": {
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "support": 0,
                },
            },
        }

    base = compute_metrics(y_true, y_pred, phishing_scores, DEFAULT_THRESHOLD)
    result = base.to_dict()
    result["support"] = base.n_samples
    result["confusion_matrix"] = base.confusion_matrix
    result["false_positive_rate"] = (
        base.false_positives / (base.true_negatives + base.false_positives)
        if base.true_negatives + base.false_positives
        else None
    )
    result["false_negative_rate"] = (
        base.false_negatives / (base.false_negatives + base.true_positives)
        if base.false_negatives + base.true_positives
        else None
    )
    has_both_classes = base.n_phishing > 0 and base.n_legitimate > 0
    result["pr_auc"] = (
        float(average_precision_score(y_true, phishing_scores)) if has_both_classes else None
    )
    result["per_class"] = {
        "legitimate": {
            "precision": base.precision_legitimate,
            "recall": base.recall_legitimate,
            "f1": base.f1_legitimate,
            "support": base.n_legitimate,
        },
        "phishing": {
            "precision": base.precision_phishing,
            "recall": base.recall_phishing,
            "f1": base.f1_phishing,
            "support": base.n_phishing,
        },
    }
    return result


def class_distribution(labels: np.ndarray) -> dict[str, Any]:
    total = int(len(labels))
    phishing = int(np.sum(labels == 1))
    legitimate = int(np.sum(labels == 0))
    return {
        "total": total,
        "phishing": phishing,
        "legitimate": legitimate,
        "phishing_fraction": phishing / total if total else None,
        "legitimate_fraction": legitimate / total if total else None,
    }


def baseline_error_rates(baseline: dict[str, Any]) -> dict[str, float | None]:
    tn = int(baseline["true_negatives"])
    fp = int(baseline["false_positives"])
    fn = int(baseline["false_negatives"])
    tp = int(baseline["true_positives"])
    return {
        "false_positive_rate": fp / (tn + fp) if tn + fp else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
    }


def compare_to_baseline(
    external: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, dict[str, float | None]]:
    baseline_values = dict(baseline)
    baseline_values.update(baseline_error_rates(baseline))
    comparison: dict[str, dict[str, float | None]] = {}
    for metric in COMPARISON_METRICS:
        external_value = external.get(metric)
        baseline_value = baseline_values.get(metric)
        delta = (
            external_value - baseline_value
            if external_value is not None and baseline_value is not None
            else None
        )
        comparison[metric] = {
            "external": external_value,
            "baseline": baseline_value,
            "delta": delta,
        }
    return comparison


def evaluate_subgroups(
    dataset: ExternalDataset,
    predictions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    https_index = FEATURE_NAMES.index("has_https")
    path_index = FEATURE_NAMES.index("path_length")
    groups = {
        "https": {
            "https": dataset.X[:, https_index] == 1,
            "http": dataset.X[:, https_index] == 0,
        },
        "url_shape": {
            "homepage": dataset.X[:, path_index] <= 1,
            "urls_with_paths": dataset.X[:, path_index] > 1,
        },
        "true_label": {
            "phishing": dataset.y == 1,
            "legitimate": dataset.y == 0,
        },
    }
    result: dict[str, Any] = {}
    for dimension, subgroups in groups.items():
        result[dimension] = {}
        for name, mask in subgroups.items():
            subset: dict[str, Any] = {
                "support": int(np.sum(mask)),
                "class_distribution": class_distribution(dataset.y[mask]),
                "models": {},
            }
            for model_name in MODEL_NAMES:
                subset["models"][model_name] = compute_external_metrics(
                    dataset.y[mask], predictions[model_name][mask], scores[model_name][mask]
                )
            result[dimension][name] = subset
    return result


def format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_report(metrics: dict[str, Any]) -> str:
    lines = [
        "=" * 78,
        "PhishGuard AI - CompPhish External Model Evaluation",
        "Evaluation threshold: 0.5 (unchanged; no tuning or calibration)",
        "Label convention: 1 = phishing (positive class), 0 = legitimate",
        "Dataset status: independent external dataset; labels are not independently verified ground truth.",
        "=" * 78,
        "",
        "1. INPUTS AND CLASS DISTRIBUTIONS",
    ]
    external = metrics["external_dataset"]
    baseline = metrics["baseline_test"]
    lines.extend(
        [
            f"  External dataset: {external['path']} ({external['sha256']})",
            f"  Baseline test set: {baseline['path']} ({baseline['sha256']})",
            f"  External support: {external['class_distribution']['total']:,}",
            (
                f"    phishing {external['class_distribution']['phishing']:,} "
                f"({external['class_distribution']['phishing_fraction']:.2%}); "
                f"legitimate {external['class_distribution']['legitimate']:,} "
                f"({external['class_distribution']['legitimate_fraction']:.2%})"
            ),
            f"  Baseline test support: {baseline['class_distribution']['total']:,}",
            (
                f"    phishing {baseline['class_distribution']['phishing']:,} "
                f"({baseline['class_distribution']['phishing_fraction']:.2%}); "
                f"legitimate {baseline['class_distribution']['legitimate']:,} "
                f"({baseline['class_distribution']['legitimate_fraction']:.2%})"
            ),
            f"  Feature columns ({len(FEATURE_NAMES)}) in model order: {', '.join(FEATURE_NAMES)}",
            "",
            "2. OVERALL METRICS",
            (
                f"  {'Metric':<28}"
                + "".join(f"{name:>22}" for name in MODEL_NAMES)
            ),
        ]
    )
    metric_rows = (
        ("Accuracy", "accuracy"),
        ("Precision (phishing)", "precision_phishing"),
        ("Recall (phishing)", "recall_phishing"),
        ("F1 (phishing)", "f1_phishing"),
        ("Precision (legitimate)", "precision_legitimate"),
        ("Recall (legitimate)", "recall_legitimate"),
        ("F1 (legitimate)", "f1_legitimate"),
        ("F1 (macro)", "f1_macro"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC (average precision)", "pr_auc"),
        ("False-positive rate", "false_positive_rate"),
        ("False-negative rate", "false_negative_rate"),
    )
    for label, key in metric_rows:
        values = "".join(
            f"{format_metric(metrics['models'][name]['overall'][key]):>22}"
            for name in MODEL_NAMES
        )
        lines.append(f"  {label:<28}{values}")
    for name in MODEL_NAMES:
        overall = metrics["models"][name]["overall"]
        lines.extend(
            [
                f"  {name} support: {overall['support']:,} "
                f"(phishing {overall['n_phishing']:,}, legitimate {overall['n_legitimate']:,})",
                f"  {name} confusion matrix [[TN, FP], [FN, TP]]: {overall['confusion_matrix']}",
                f"  {name} per-class metrics (precision / recall / F1 / support):",
            ]
        )
        for class_name in ("phishing", "legitimate"):
            class_metrics = overall["per_class"][class_name]
            lines.append(
                f"    {class_name}: {format_metric(class_metrics['precision'])} / "
                f"{format_metric(class_metrics['recall'])} / "
                f"{format_metric(class_metrics['f1'])} / {class_metrics['support']:,}"
            )

    lines.extend(["", "3. SUBGROUP METRICS"])
    subgroup_labels = {
        "https": {"https": "HTTPS", "http": "HTTP"},
        "url_shape": {
            "homepage": "Homepage/root (path_length <= 1)",
            "urls_with_paths": "URLs with paths (path_length > 1)",
        },
        "true_label": {"phishing": "Phishing labels", "legitimate": "Legitimate labels"},
    }
    for dimension, subgroups in metrics["subgroups"].items():
        lines.append(f"  {dimension.replace('_', ' ').title()}:")
        for subgroup_name, group_label in subgroup_labels[dimension].items():
            subgroup = subgroups[subgroup_name]
            lines.append(
                f"    {group_label}: support {subgroup['support']:,}; "
                f"phishing {subgroup['class_distribution']['phishing']:,}; "
                f"legitimate {subgroup['class_distribution']['legitimate']:,}"
            )
            for model_name in MODEL_NAMES:
                result = subgroup["models"][model_name]
                lines.append(
                    f"      {model_name}: accuracy {format_metric(result['accuracy'])}, "
                    f"F1 phishing {format_metric(result['f1_phishing'])}, "
                    f"ROC-AUC {format_metric(result['roc_auc'])}, "
                    f"PR-AUC {format_metric(result['pr_auc'])}, "
                    f"FPR {format_metric(result['false_positive_rate'])}, "
                    f"FNR {format_metric(result['false_negative_rate'])}"
                )

    lines.extend(
        [
            "",
            "4. EXTERNAL RESULTS VS. ORIGINAL BASELINE TEST RESULTS",
            "  Delta is external minus baseline. PR-AUC was not saved in the baseline metrics.",
        ]
    )
    for name in MODEL_NAMES:
        lines.append(f"  {name}:")
        for metric in COMPARISON_METRICS:
            values = metrics["models"][name]["comparison_to_baseline"][metric]
            delta_text = format_metric(values["delta"])
            lines.append(
                f"    {metric:<25} external {format_metric(values['external']):>8} "
                f"baseline {format_metric(values['baseline']):>8} delta {delta_text:>8}"
            )

    lines.extend(
        [
            "",
            "5. INTERPRETATION AND LIMITATIONS",
            "  CompPhish is treated as an independent external evaluation dataset, but its",
            "  labels have not been independently verified as ground truth.",
            "  Poor external performance indicates limited generalization to this dataset;",
            "  it does not, by itself, establish a model implementation defect.",
            "  PR-AUC is average precision; ROC-AUC and PR-AUC are undefined for a",
            "  single-class subgroup. Error rates with a zero denominator are reported n/a.",
            "  URL structure subgroups use the existing path_length feature: <= 1 is",
            "  homepage/root and > 1 is a URL with a path.",
            "  Models, labels, feature columns, and the 0.5 decision threshold were not",
            "  retrained, tuned, calibrated, or altered. URLs were not fetched.",
            "",
            "Artifacts:",
            f"  {metrics['artifacts']['metrics_json']}",
            f"  {metrics['artifacts']['report_text']}",
            f"  {metrics['artifacts']['predictions_csv']}",
            "=" * 78,
        ]
    )
    return "\n".join(lines) + "\n"


def write_predictions(
    path: Path,
    dataset: ExternalDataset,
    scores: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for index in range(len(dataset.y)):
            row: dict[str, Any] = {
                "url": dataset.urls[index],
                "domain": dataset.domains[index],
                "true_label": int(dataset.y[index]),
                "source": dataset.sources[index],
                "source_record_id": dataset.source_record_ids[index],
            }
            for name in MODEL_NAMES:
                row[f"{name}_score"] = repr(float(scores[name][index]))
                row[f"{name}_predicted_label"] = int(predictions[name][index])
            writer.writerow(row)


def run_evaluation(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    metadata_path = project_root / MODEL_METADATA_PATH
    metadata = load_json_object(metadata_path, "model metadata")
    validate_model_metadata(metadata)

    baseline_path = project_root / BASELINE_METRICS_PATH
    baseline_document = load_json_object(baseline_path, "baseline metrics")
    baseline_test = validate_baseline_metrics(baseline_document)

    dataset_path = project_root / EXTERNAL_DATASET_PATH
    dataset = load_external_dataset(dataset_path)
    models = load_models(project_root)

    scores: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    overall: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        scores[name] = model_phishing_scores(models[name], dataset.X, name)
        predictions[name] = (scores[name] >= DEFAULT_THRESHOLD).astype(np.int64)
        overall[name] = compute_external_metrics(dataset.y, predictions[name], scores[name])

    subgroups = evaluate_subgroups(dataset, predictions, scores)
    baseline_test_counts = baseline_test[RANDOM_FOREST]
    baseline_distribution = {
        "total": int(baseline_test_counts["n_samples"]),
        "phishing": int(baseline_test_counts["n_phishing"]),
        "legitimate": int(baseline_test_counts["n_legitimate"]),
    }
    baseline_distribution["phishing_fraction"] = (
        baseline_distribution["phishing"] / baseline_distribution["total"]
    )
    baseline_distribution["legitimate_fraction"] = (
        baseline_distribution["legitimate"] / baseline_distribution["total"]
    )

    output_paths = {
        "metrics_json": "reports/evaluation/compphish_external_metrics.json",
        "report_text": "reports/evaluation/compphish_external_report.txt",
        "predictions_csv": "reports/evaluation/compphish_external_predictions.csv",
    }
    document: dict[str, Any] = {
        "evaluation": {
            "label_convention": {"1": "phishing", "0": "legitimate"},
            "positive_class": 1,
            "threshold": DEFAULT_THRESHOLD,
            "feature_names": list(FEATURE_NAMES),
            "pr_auc_definition": "average_precision",
        },
        "external_dataset": {
            "path": EXTERNAL_DATASET_PATH.as_posix(),
            "sha256": sha256_of(dataset_path),
            "class_distribution": class_distribution(dataset.y),
        },
        "baseline_test": {
            "path": "reports/training/baseline_metrics.json",
            "sha256": sha256_of(baseline_path),
            "dataset_sha256": baseline_document.get("dataset_sha256"),
            "class_distribution": baseline_distribution,
            "models": baseline_test,
        },
        "model_metadata": {
            "path": "models/baseline/model_metadata.json",
            "sha256": sha256_of(metadata_path),
        },
        "model_files": {
            name: {
                "path": (MODELS_DIR / f"{name}.joblib").as_posix(),
                "sha256": sha256_of(project_root / MODELS_DIR / f"{name}.joblib"),
            }
            for name in MODEL_NAMES
        },
        "models": {
            name: {
                "overall": overall[name],
                "comparison_to_baseline": compare_to_baseline(overall[name], baseline_test[name]),
            }
            for name in MODEL_NAMES
        },
        "subgroups": subgroups,
        "artifacts": output_paths,
    }

    output_dir = project_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = project_root / output_paths["metrics_json"]
    report_path = project_root / output_paths["report_text"]
    predictions_path = project_root / output_paths["predictions_csv"]
    metrics_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(format_report(document), encoding="utf-8")
    write_predictions(predictions_path, dataset, scores, predictions)
    return document


def main() -> None:
    metrics = run_evaluation()
    for name in MODEL_NAMES:
        overall = metrics["models"][name]["overall"]
        print(
            f"{name}: accuracy={overall['accuracy']:.4f}, "
            f"F1(phishing)={overall['f1_phishing']:.4f}, "
            f"ROC-AUC={format_metric(overall['roc_auc'])}, "
            f"PR-AUC={format_metric(overall['pr_auc'])}"
        )
    print(f"Metrics: {metrics['artifacts']['metrics_json']}")
    print(f"Report: {metrics['artifacts']['report_text']}")
    print(f"Predictions: {metrics['artifacts']['predictions_csv']}")


if __name__ == "__main__":
    main()
