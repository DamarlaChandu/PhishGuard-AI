"""Train and evaluate the PhishGuard AI baseline models.

Input (read-only):  data/processed/phishing_dataset.csv
Outputs:
  models/baseline/random_forest.joblib
  models/baseline/logistic_regression.joblib
  models/baseline/model_metadata.json
  reports/training/baseline_metrics.json
  reports/training/random_forest_feature_importance.csv
  reports/training/baseline_training_report.txt

Model inputs are the 15 backend URL features only. The split groups rows by
registrable domain so no host (or sibling subdomain) is in both train and
test. Tranco is not used. The dataset file is verified unchanged afterwards.

Requires scikit-learn, numpy and joblib (see backend/requirements-ml.txt).

Usage (from the project root):
    python scripts/train_baseline_models.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from app.ml.features import FEATURE_NAMES  # noqa: E402
from app.ml.training.baseline_models import (  # noqa: E402
    DEFAULT_SEED,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    build_baseline_models,
    build_logistic_regression,
    build_random_forest,
)
from app.ml.training.evaluation import (  # noqa: E402
    SHORTCUT_SEPARATION_THRESHOLD,
    EvaluationResult,
    evaluate_classifier,
    logistic_regression_coefficients,
    random_forest_feature_importance,
    single_feature_roc_auc,
)
from app.ml.training.scheme_bias import (  # noqa: E402
    assess_scheme_bias,
    run_scheme_ablation,
    summarize_scheme_bias,
)
from app.ml.training.training_data import (  # noqa: E402
    METADATA_COLUMNS,
    TARGET_COLUMN,
    group_stratified_split,
    load_feature_dataset,
    overlapping_values,
)

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phishing_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "baseline"
REPORTS_DIR = PROJECT_ROOT / "reports" / "training"

SEED = DEFAULT_SEED
TEST_FRACTION = 0.2
CLASS_SHARE_TOLERANCE = 0.02  # warn if test phishing share differs more than this
OVERFIT_GAP_WARNING = 0.05  # warn if train - test F1 exceeds this

MODEL_BUILDERS = {RANDOM_FOREST: build_random_forest, LOGISTIC_REGRESSION: build_logistic_regression}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def class_line(y: np.ndarray) -> str:
    n = len(y)
    phishing = int((y == 1).sum())
    return f"{n:>7,} rows | phishing {phishing:>6,} ({phishing / n:.2%}) | legitimate {n - phishing:>6,} ({(n - phishing) / n:.2%})"


def metrics_table(results: dict[str, EvaluationResult]) -> list[str]:
    rows = [
        ("Accuracy", "accuracy"),
        ("Precision (phishing)", "precision_phishing"),
        ("Recall (phishing)", "recall_phishing"),
        ("F1 (phishing)", "f1_phishing"),
        ("Precision (legitimate)", "precision_legitimate"),
        ("Recall (legitimate)", "recall_legitimate"),
        ("F1 (legitimate)", "f1_legitimate"),
        ("F1 (macro)", "f1_macro"),
        ("ROC-AUC", "roc_auc"),
    ]
    names = list(results)
    lines = [f"  {'Metric':<26}" + "".join(f"{n:>22}" for n in names)]
    for label, attr in rows:
        lines.append(f"  {label:<26}" + "".join(f"{fmt(getattr(results[n], attr)):>22}" for n in names))
    return lines


def confusion_lines(result: EvaluationResult) -> list[str]:
    return [
        "                         predicted legit   predicted phishing",
        f"    actual legitimate  {result.true_negatives:>15,} {result.false_positives:>20,}",
        f"    actual phishing    {result.false_negatives:>15,} {result.true_positives:>20,}",
    ]


def main() -> None:
    started = datetime.now(timezone.utc)
    dataset_hash_before = sha256_of(DATASET_PATH)
    pipeline_warnings: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # ---------------- Data and split ----------------
        dataset = load_feature_dataset(DATASET_PATH)
        split = group_stratified_split(dataset.y, dataset.groups, TEST_FRACTION, SEED)
        train, test = dataset.subset(split.train), dataset.subset(split.test)

        host_overlap = overlapping_values(train.hostnames, test.hostnames)
        group_overlap = overlapping_values(train.groups, test.groups)
        if host_overlap or group_overlap:
            raise SystemExit(
                f"Leakage: {len(host_overlap)} hosts / {len(group_overlap)} groups in both splits"
            )

        overall_share = dataset.y.mean()
        for name, part in (("train", train), ("test", test)):
            if abs(part.y.mean() - overall_share) > CLASS_SHARE_TOLERANCE:
                pipeline_warnings.append(
                    f"{name} phishing share {part.y.mean():.2%} differs from overall "
                    f"{overall_share:.2%} by more than {CLASS_SHARE_TOLERANCE:.0%}"
                )

        group_labels: dict[str, set[int]] = {}
        for group, label in zip(dataset.groups.tolist(), dataset.y.tolist()):
            group_labels.setdefault(group, set()).add(label)
        mixed_groups = sorted(g for g, labels in group_labels.items() if len(labels) > 1)
        group_sizes = Counter(dataset.groups.tolist())

        # ---------------- Training and evaluation ----------------
        models = build_baseline_models(SEED)
        fit_seconds: dict[str, float] = {}
        test_results: dict[str, EvaluationResult] = {}
        train_results: dict[str, EvaluationResult] = {}
        for name, model in models.items():
            start = time.perf_counter()
            model.fit(train.X, train.y)
            fit_seconds[name] = time.perf_counter() - start
            test_results[name] = evaluate_classifier(model, test.X, test.y)
            train_results[name] = evaluate_classifier(model, train.X, train.y)
            gap = train_results[name].f1_phishing - test_results[name].f1_phishing
            if gap > OVERFIT_GAP_WARNING:
                pipeline_warnings.append(
                    f"{name}: train F1 exceeds test F1 by {gap:.4f} (possible overfitting)"
                )

        importances = random_forest_feature_importance(
            models[RANDOM_FOREST], test.X, test.y, FEATURE_NAMES, seed=SEED
        )
        coefficients = logistic_regression_coefficients(models[LOGISTIC_REGRESSION], FEATURE_NAMES)

        # ---------------- Missing-scheme bias ----------------
        https_index = FEATURE_NAMES.index("has_https")
        bias_summary = summarize_scheme_bias(dataset, https_index)
        ablations = [
            run_scheme_ablation(
                name, models[name], lambda builder=MODEL_BUILDERS[name]: builder(SEED), train, test
            )
            for name in models
        ]
        bias_verdict = assess_scheme_bias(bias_summary, ablations)
        if bias_verdict.startswith(("STRONG", "WEAK")):
            pipeline_warnings.append(f"Missing-scheme artifact: {bias_verdict.split(':')[0]}")

        # ---------------- Shortcut / artifact check ----------------
        single_scores = single_feature_roc_auc(test.X, test.y, FEATURE_NAMES)
        shortcuts = [s for s in single_scores if s.is_shortcut]
        for score in shortcuts:
            pipeline_warnings.append(
                f"Shortcut feature: {score.feature} alone separates the test classes "
                f"(ROC-AUC {score.roc_auc:.4f}); test metrics are likely inflated by a "
                "dataset-collection artifact"
            )
        path_index = FEATURE_NAMES.index("path_length")
        subgroups = {
            "legitimate URLs with a path (path_length > 1)": (test.y == 0) & (test.X[:, path_index] > 1),
            "legitimate URLs over http": (test.y == 0) & (test.X[:, https_index] == 0),
            "phishing URLs over https": (test.y == 1) & (test.X[:, https_index] == 1),
            "phishing URLs from OpenPhish (current feed)": (test.y == 1) & (test.sources == "openphish"),
        }
        subgroup_results: dict[str, dict[str, EvaluationResult]] = {}
        for description, mask in subgroups.items():
            if mask.any():
                subgroup_results[description] = {
                    name: evaluate_classifier(model, test.X[mask], test.y[mask])
                    for name, model in models.items()
                }
        for description, by_model in subgroup_results.items():
            for name, result in by_model.items():
                correct = result.true_negatives + result.true_positives
                if result.n_samples and correct / result.n_samples < 0.8:
                    pipeline_warnings.append(
                        f"{name}: only {correct / result.n_samples:.1%} correct on test "
                        f"{description} (n={result.n_samples})"
                    )
        path_stats = {
            label: {
                "empty_or_root_path": float(np.mean(dataset.X[dataset.y == label, path_index] <= 1)),
                "longer_path": float(np.mean(dataset.X[dataset.y == label, path_index] > 1)),
            }
            for label in (0, 1)
        }

    python_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    # ---------------- Save artifacts ----------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = {}
    for name, model in models.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        model_paths[name] = rel(path)

    environment = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    split_info = {
        "method": "StratifiedGroupKFold (one fold as test), groups = registrable domain",
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_phishing": int(train.y.sum()),
        "test_phishing": int(test.y.sum()),
        "train_groups": len(set(train.groups.tolist())),
        "test_groups": len(set(test.groups.tolist())),
        "host_overlap": len(host_overlap),
        "group_overlap": len(group_overlap),
    }
    metadata = {
        "created_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label_convention": {"1": "phishing", "0": "legitimate"},
        "positive_class": 1,
        "decision_threshold": 0.5,
        "feature_names": list(FEATURE_NAMES),
        "excluded_columns": list(METADATA_COLUMNS),
        "target_column": TARGET_COLUMN,
        "dataset": {"path": rel(DATASET_PATH), "sha256": dataset_hash_before, "rows": len(dataset)},
        "split": split_info,
        "models": model_paths,
        "environment": environment,
        "note": "joblib files are pickles: load only files produced by this project.",
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_json = {
        "created_at": metadata["created_at"],
        "dataset_sha256": dataset_hash_before,
        "split": split_info,
        "test": {name: r.to_dict() for name, r in test_results.items()},
        "train": {name: r.to_dict() for name, r in train_results.items()},
        "fit_seconds": fit_seconds,
        "random_forest_feature_importance": [vars(i) for i in importances],
        "logistic_regression_coefficients": dict(coefficients),
        "scheme_bias": {
            "summary": bias_summary.to_dict(),
            "ablations": [
                {
                    "model": a.model_name,
                    "train_flagged_rows": a.train_flagged_rows,
                    "test_flagged_rows": a.test_flagged_rows,
                    "with_flagged_on_clean_test": a.with_flagged_on_clean_test.to_dict(),
                    "without_flagged_on_clean_test": a.without_flagged_on_clean_test.to_dict(),
                    "with_flagged_on_flagged_test": (
                        a.with_flagged_on_flagged_test.to_dict()
                        if a.with_flagged_on_flagged_test else None
                    ),
                    "f1_delta": a.f1_delta,
                    "roc_auc_delta": a.roc_auc_delta,
                }
                for a in ablations
            ],
            "verdict": bias_verdict,
        },
        "shortcut_check": {
            "separation_threshold": SHORTCUT_SEPARATION_THRESHOLD,
            "single_feature_test_roc_auc": {s.feature: s.roc_auc for s in single_scores},
            "shortcut_features": [s.feature for s in shortcuts],
            "path_length_by_label": path_stats,
            "subgroups": {
                description: {name: r.to_dict() for name, r in by_model.items()}
                for description, by_model in subgroup_results.items()
            },
        },
        "warnings": {"pipeline": pipeline_warnings, "python": python_warnings},
        "environment": environment,
    }
    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    importance_path = REPORTS_DIR / "random_forest_feature_importance.csv"
    with importance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature", "permutation_importance_mean", "permutation_importance_std", "impurity_importance"])
        for rank, item in enumerate(importances, start=1):
            writer.writerow([rank, item.feature, f"{item.permutation_mean:.6f}", f"{item.permutation_std:.6f}", f"{item.impurity_importance:.6f}"])

    dataset_hash_after = sha256_of(DATASET_PATH)
    if dataset_hash_after != dataset_hash_before:
        raise SystemExit("Dataset file changed during training")

    # ---------------- Report ----------------
    out: list[str] = []
    add = out.append
    rule = "=" * 78
    add(rule)
    add("PhishGuard AI - Baseline Model Training Report")
    add(f"Generated: {started.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    add("Generator: scripts/train_baseline_models.py")
    add("Label convention: 1 = phishing (positive class), 0 = legitimate; threshold 0.5")
    add(rule)

    add("\n1. DATA")
    add(f"  Dataset:  {rel(DATASET_PATH)}")
    add(f"  SHA-256:  {dataset_hash_before} (unchanged after run)")
    add(f"  All:     {class_line(dataset.y)}")
    add(f"  Model inputs ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    add(f"  Excluded from inputs: {', '.join(METADATA_COLUMNS)} (target: {TARGET_COLUMN})")
    add("  Tranco: not used.")

    add("\n2. SPLIT (grouped by registrable domain, stratified, seed 42)")
    add(f"  Train:   {class_line(train.y)}")
    add(f"  Test:    {class_line(test.y)}")
    add(f"  Domain groups: {len(group_sizes):,} total | train {split_info['train_groups']:,} | test {split_info['test_groups']:,}")
    add(f"  Distinct hostnames: train {len(set(train.hostnames.tolist())):,} | test {len(set(test.hostnames.tolist())):,}")
    add(f"  Hostnames in both train and test: {len(host_overlap)}")
    add(f"  Domain groups in both train and test: {len(group_overlap)}")
    add(f"  Groups containing both labels (kept whole on one side): {len(mixed_groups):,}")
    for group in sorted(mixed_groups, key=lambda g: -group_sizes[g])[:8]:
        side = "test" if group in set(test.groups.tolist()) else "train"
        add(f"    {group}: {group_sizes[group]:,} rows ({side})")
    add("  Largest groups:")
    test_groups = set(test.groups.tolist())
    for group, size in group_sizes.most_common(6):
        add(f"    {group}: {size:,} rows ({'test' if group in test_groups else 'train'})")

    add("\n3. TEST-SET METRICS")
    out.extend(metrics_table(test_results))
    for name, result in test_results.items():
        add(f"  Confusion matrix - {name} (test, {result.n_samples:,} rows):")
        out.extend(confusion_lines(result))

    add("\n  Train-set metrics (for overfitting check):")
    out.extend(metrics_table(train_results))
    add("  Fit time: " + ", ".join(f"{n} {s:.1f}s" for n, s in fit_seconds.items()))

    add("\n4. RANDOM FOREST FEATURE IMPORTANCE")
    add("  Permutation importance = mean drop in test ROC-AUC when the feature is")
    add("  shuffled (5 repeats). Impurity importance comes from training splits.")
    add(f"  {'#':>3} {'feature':<26}{'permutation':>14}{'± std':>10}{'impurity':>12}")
    for rank, item in enumerate(importances, start=1):
        add(f"  {rank:>3} {item.feature:<26}{item.permutation_mean:>14.4f}{item.permutation_std:>10.4f}{item.impurity_importance:>12.4f}")
    add("\n  Logistic Regression coefficients (after log1p + standardization;")
    add("  positive pushes toward phishing):")
    for name, coefficient in coefficients:
        add(f"    {name:<26}{coefficient:>+10.4f}")

    s = bias_summary
    add("\n5. MISSING-SCHEME (added_default_scheme) BIAS CHECK")
    add(f"  Rows with a default 'http://' added: {s.flagged_rows:,} "
        f"(phishing {s.flagged_phishing:,}, legitimate {s.flagged_legitimate:,})")
    add(f"  Phishing rows with has_https = 0:    {s.phishing_http_rows:,}")
    add(f"  Flagged share of those:              {fmt(s.flagged_share_of_phishing_http)}")
    add(f"  P(phishing | has_https = 0):         {fmt(s.phishing_rate_among_http)} "
        f"(excluding flagged: {fmt(s.phishing_rate_among_http_excluding_flagged)})")
    add(f"  has_https rate, phishing:            {fmt(s.https_rate_phishing)} "
        f"(excluding flagged: {fmt(s.https_rate_phishing_excluding_flagged)})")
    add(f"  has_https rate, legitimate:          {fmt(s.https_rate_legitimate)}")
    add("  Ablation: each model retrained WITHOUT flagged training rows; both")
    add("  versions scored on test rows that are not flagged:")
    for a in ablations:
        w, wo = a.with_flagged_on_clean_test, a.without_flagged_on_clean_test
        add(f"    {a.model_name}: train flagged {a.train_flagged_rows:,}, test flagged {a.test_flagged_rows:,}")
        add(f"      F1 (phishing)  with {w.f1_phishing:.4f}  without {wo.f1_phishing:.4f}  delta {a.f1_delta:+.4f}")
        add(f"      ROC-AUC        with {fmt(w.roc_auc)}  without {fmt(wo.roc_auc)}  delta {fmt(a.roc_auc_delta)}")
        if a.with_flagged_on_flagged_test is not None:
            f = a.with_flagged_on_flagged_test
            add(f"      Recall on the {f.n_samples:,} flagged test rows (all phishing): {f.recall_phishing:.4f}")
    add(f"  Verdict: {bias_verdict}")
    add("  No rows were removed from the dataset or the saved models.")

    add("\n6. SHORTCUT / DATASET-ARTIFACT CHECK")
    add("  Each raw feature used alone as a score on the test set. Separation =")
    add(f"  max(AUC, 1 - AUC); >= {SHORTCUT_SEPARATION_THRESHOLD} means the feature alone nearly separates")
    add("  the classes, which usually reflects how the data was collected.")
    add(f"  {'feature':<26}{'ROC-AUC':>10}{'separation':>12}")
    for score in single_scores:
        flag = "  <-- shortcut" if score.is_shortcut else ""
        add(f"  {score.feature:<26}{score.roc_auc:>10.4f}{score.separation:>12.4f}{flag}")
    add("  path_length by label (whole dataset):")
    for label, name in ((1, "phishing"), (0, "legitimate")):
        stats = path_stats[label]
        add(f"    {name:<11} empty or '/' path {stats['empty_or_root_path']:.2%} | longer path {stats['longer_path']:.2%}")
    add("  Test-set subgroups (fraction classified correctly):")
    for description, by_model in subgroup_results.items():
        n = next(iter(by_model.values())).n_samples
        parts = []
        for name, result in by_model.items():
            correct = (result.true_negatives + result.true_positives) / result.n_samples
            parts.append(f"{name} {correct:.1%}")
        add(f"    {description} (n={n:,}): " + ", ".join(parts))
    if shortcuts:
        add("  CONCLUSION: the headline test metrics are inflated by dataset artifacts.")
        add("  The legitimate class is dominated by bare https homepages and the")
        add("  phishing class by http URLs with paths, so the models largely learn")
        add("  'has a path / no https -> phishing'. Do not treat these baselines as")
        add("  deployable detectors; the dataset needs legitimate URLs with paths")
        add("  (and more https phishing) before metrics are meaningful.")

    add("\n7. WARNINGS")
    if not pipeline_warnings and not python_warnings:
        add("  none")
    for message in pipeline_warnings:
        add(f"  [pipeline] {message}")
    for message in python_warnings:
        add(f"  [python]   {message}")

    add("\n8. ARTIFACTS")
    for path in model_paths.values():
        add(f"  {path}")
    add(f"  {rel(MODELS_DIR / 'model_metadata.json')}")
    add(f"  {rel(metrics_path)}")
    add(f"  {rel(importance_path)}")
    add(f"  {rel(REPORTS_DIR / 'baseline_training_report.txt')}")
    add("  Environment: " + ", ".join(f"{k} {v}" for k, v in environment.items()))
    add("\n" + rule)

    report = "\n".join(out) + "\n"
    (REPORTS_DIR / "baseline_training_report.txt").write_text(report, encoding="utf-8")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    stdout.write(report)
    stdout.flush()


if __name__ == "__main__":
    main()
