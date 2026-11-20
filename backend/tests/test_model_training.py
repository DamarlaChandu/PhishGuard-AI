"""Tests for training data loading, grouped splitting, models, metrics and bias checks.

Skipped when scikit-learn is not installed (see requirements-ml.txt).
"""

import csv
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from app.ml.features import FEATURE_NAMES  # noqa: E402
from app.ml.training.baseline_models import (  # noqa: E402
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    build_baseline_models,
    build_logistic_regression,
    build_random_forest,
)
from app.ml.training.dataset_preparation import OUTPUT_COLUMNS  # noqa: E402
from app.ml.training.evaluation import (  # noqa: E402
    compute_metrics,
    evaluate_classifier,
    logistic_regression_coefficients,
    random_forest_feature_importance,
    single_feature_roc_auc,
)
from app.ml.training.scheme_bias import (  # noqa: E402
    SchemeAblation,
    assess_scheme_bias,
    run_scheme_ablation,
    scheme_added_mask,
    summarize_scheme_bias,
)
from app.ml.training.training_data import (  # noqa: E402
    FeatureDataset,
    group_stratified_split,
    load_feature_dataset,
    overlapping_values,
    registrable_domain,
)

N_FEATURES = len(FEATURE_NAMES)
HTTPS = FEATURE_NAMES.index("has_https")
PATH = FEATURE_NAMES.index("path_length")


def make_dataset(
    n: int = 200,
    seed: int = 0,
    actions: list[str] | None = None,
    hosts_per_group: int = 2,
) -> FeatureDataset:
    """Synthetic dataset where path_length carries the signal."""
    rng = np.random.default_rng(seed)
    y = np.array([1 if i % 3 else 0 for i in range(n)], dtype=np.int64)
    X = rng.integers(0, 5, size=(n, N_FEATURES)).astype(np.float64)
    X[:, PATH] = np.where(y == 1, rng.integers(5, 40, n), rng.integers(0, 3, n))
    X[:, HTTPS] = np.where(y == 1, 0, 1)
    hosts = np.array([f"h{i % (n // hosts_per_group)}.site{i % (n // 4)}.com" for i in range(n)])
    return FeatureDataset(
        X=X,
        y=y,
        hostnames=hosts,
        groups=np.array([registrable_domain(h) for h in hosts]),
        sources=np.array(["legitphish"] * n),
        cleaning_actions=np.array(actions if actions is not None else [""] * n),
    )


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def csv_row(url: str, domain: str, label: int, **features: int) -> dict:
    row = {
        "url": url,
        "domain": domain,
        "label": label,
        "source": "legitphish",
        "collected_at": "2026-01-01T00:00:00Z",
        "cleaning_actions": "",
    }
    row.update({name: features.get(name, 0) for name in FEATURE_NAMES})
    return row


class TestRegistrableDomain:
    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("example.com", "example.com"),
            ("www.example.com", "example.com"),
            ("a.b.example.com", "example.com"),
            ("shop.example.co.uk", "example.co.uk"),
            ("192.0.2.1", "192.0.2.1"),
            ("2001:db8::1", "2001:db8::1"),
            ("localhost", "localhost"),
        ],
    )
    def test_registrable_domain(self, host: str, expected: str) -> None:
        assert registrable_domain(host) == expected


class TestLoadFeatureDataset:
    def test_loads_features_in_order_and_excludes_metadata(self, tmp_path: Path) -> None:
        features = {name: i + 1 for i, name in enumerate(FEATURE_NAMES)}
        path = write_csv(
            tmp_path / "d.csv",
            [
                csv_row("https://a.example.com/x", "a.example.com", 1, **features),
                csv_row("https://example.org", "example.org", 0),
            ],
        )

        dataset = load_feature_dataset(path)

        assert dataset.X.shape == (2, N_FEATURES)
        assert dataset.X[0].tolist() == [float(i + 1) for i in range(N_FEATURES)]
        assert dataset.y.tolist() == [1, 0]
        assert dataset.groups.tolist() == ["example.com", "example.org"]
        assert dataset.feature_names == FEATURE_NAMES

    def test_column_order_in_file_does_not_matter(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        columns = list(reversed(OUTPUT_COLUMNS))
        row = csv_row("https://e.com/", "e.com", 1, url_length=14)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(row)

        dataset = load_feature_dataset(path)

        assert dataset.X[0, FEATURE_NAMES.index("url_length")] == 14

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        path.write_text("url,label\nhttps://e.com,1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="missing columns"):
            load_feature_dataset(path)

    @pytest.mark.parametrize("label", [2, -1])
    def test_invalid_label_raises(self, tmp_path: Path, label: int) -> None:
        path = write_csv(tmp_path / "d.csv", [csv_row("https://e.com", "e.com", label)])

        with pytest.raises(ValueError, match="label must be 0 or 1"):
            load_feature_dataset(path)

    def test_non_integer_feature_raises(self, tmp_path: Path) -> None:
        row = csv_row("https://e.com", "e.com", 1)
        row["url_length"] = "abc"
        path = write_csv(tmp_path / "d.csv", [row])

        with pytest.raises(ValueError, match="url_length must be an integer"):
            load_feature_dataset(path)

    def test_negative_feature_raises(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "d.csv", [csv_row("https://e.com", "e.com", 1, num_dots=-1)])

        with pytest.raises(ValueError, match="non-negative"):
            load_feature_dataset(path)

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "d.csv", [])

        with pytest.raises(ValueError, match="no rows"):
            load_feature_dataset(path)


class TestGroupSplit:
    def test_no_group_or_host_in_both_splits(self) -> None:
        dataset = make_dataset(400)
        split = group_stratified_split(dataset.y, dataset.groups, 0.2, seed=1)

        train, test = dataset.subset(split.train), dataset.subset(split.test)
        assert overlapping_values(train.groups, test.groups) == set()
        assert overlapping_values(train.hostnames, test.hostnames) == set()

    def test_split_covers_every_row_exactly_once(self) -> None:
        dataset = make_dataset(400)
        split = group_stratified_split(dataset.y, dataset.groups)

        combined = np.concatenate([split.train, split.test])
        assert sorted(combined.tolist()) == list(range(len(dataset)))

    def test_split_is_reproducible(self) -> None:
        dataset = make_dataset(400)
        a = group_stratified_split(dataset.y, dataset.groups, seed=7)
        b = group_stratified_split(dataset.y, dataset.groups, seed=7)

        assert a.test.tolist() == b.test.tolist()

    def test_different_seed_changes_split(self) -> None:
        dataset = make_dataset(400)
        a = group_stratified_split(dataset.y, dataset.groups, seed=1)
        b = group_stratified_split(dataset.y, dataset.groups, seed=2)

        assert a.test.tolist() != b.test.tolist()

    def test_test_size_and_class_balance_are_close(self) -> None:
        dataset = make_dataset(1000)
        split = group_stratified_split(dataset.y, dataset.groups, 0.2)

        assert 0.15 <= len(split.test) / len(dataset) <= 0.25
        assert abs(dataset.y[split.test].mean() - dataset.y.mean()) < 0.05

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 0.7, -0.1])
    def test_invalid_fraction_raises(self, fraction: float) -> None:
        dataset = make_dataset(100)
        with pytest.raises(ValueError, match="test_fraction"):
            group_stratified_split(dataset.y, dataset.groups, fraction)

    def test_too_few_groups_raises(self) -> None:
        y = np.array([0, 1, 0, 1])
        groups = np.array(["a", "a", "b", "b"])
        with pytest.raises(ValueError, match="distinct groups"):
            group_stratified_split(y, groups, 0.2)


class TestModels:
    def test_baseline_models_are_configured(self) -> None:
        models = build_baseline_models(seed=3)

        assert set(models) == {RANDOM_FOREST, LOGISTIC_REGRESSION}
        assert models[RANDOM_FOREST].random_state == 3
        assert models[RANDOM_FOREST].class_weight == "balanced"
        steps = [name for name, _ in models[LOGISTIC_REGRESSION].steps]
        assert steps == ["log1p", "scale", "classifier"]

    def test_logistic_regression_scaling_uses_training_data_only(self) -> None:
        dataset = make_dataset(300)
        model = build_logistic_regression().fit(dataset.X, dataset.y)

        transformed = model[:-1].transform(dataset.X)
        assert np.allclose(transformed.mean(axis=0), 0, atol=1e-8)

    @pytest.mark.parametrize("builder", [build_random_forest, build_logistic_regression])
    def test_models_learn_a_clear_signal(self, builder) -> None:
        train, test = make_dataset(400, seed=1), make_dataset(200, seed=2)
        model = builder().fit(train.X, train.y)

        result = evaluate_classifier(model, test.X, test.y)
        assert result.accuracy > 0.95
        assert result.roc_auc is not None and result.roc_auc > 0.95


class TestMetrics:
    def test_compute_metrics_known_values(self) -> None:
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.4, 0.1, 0.2, 0.3, 0.6])

        result = compute_metrics(y_true, y_pred, scores)

        assert result.confusion_matrix == [[3, 1], [1, 3]]
        assert result.accuracy == pytest.approx(0.75)
        assert result.precision_phishing == pytest.approx(0.75)
        assert result.recall_phishing == pytest.approx(0.75)
        assert result.f1_phishing == pytest.approx(0.75)
        assert result.recall_legitimate == pytest.approx(0.75)
        # 15 of 16 (phishing, legitimate) score pairs are ordered correctly.
        assert result.roc_auc == pytest.approx(15 / 16)
        assert (result.n_phishing, result.n_legitimate) == (4, 4)

    def test_roc_auc_is_none_with_one_class(self) -> None:
        result = compute_metrics(np.array([1, 1]), np.array([1, 0]), np.array([0.9, 0.1]))

        assert result.roc_auc is None
        assert result.recall_phishing == pytest.approx(0.5)
        assert result.precision_legitimate == 0.0  # zero_division handled

    def test_evaluate_uses_phishing_column_regardless_of_class_order(self) -> None:
        class ReversedClasses:
            classes_ = np.array([1, 0])

            def predict_proba(self, X):
                # column 0 is phishing because classes_ = [1, 0]
                return np.column_stack([X[:, 0], 1 - X[:, 0]])

        X = np.array([[0.9], [0.8], [0.2], [0.1]])
        y = np.array([1, 1, 0, 0])

        result = evaluate_classifier(ReversedClasses(), X, y)

        assert result.accuracy == 1.0
        assert result.roc_auc == 1.0

    def test_threshold_changes_predictions(self) -> None:
        class Fixed:
            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                return np.column_stack([1 - X[:, 0], X[:, 0]])

        X = np.array([[0.6], [0.4]])
        y = np.array([1, 0])

        assert evaluate_classifier(Fixed(), X, y, threshold=0.5).accuracy == 1.0
        assert evaluate_classifier(Fixed(), X, y, threshold=0.7).recall_phishing == 0.0

    def test_empty_evaluation_raises(self) -> None:
        dataset = make_dataset(60)
        model = build_random_forest().fit(dataset.X, dataset.y)
        with pytest.raises(ValueError, match="empty"):
            evaluate_classifier(model, np.empty((0, N_FEATURES)), np.empty(0))


class TestFeatureDiagnostics:
    def test_single_feature_auc_flags_separating_features(self) -> None:
        dataset = make_dataset(300)
        scores = {s.feature: s for s in single_feature_roc_auc(dataset.X, dataset.y, FEATURE_NAMES)}

        assert scores["path_length"].is_shortcut  # perfect positive separation
        assert scores["has_https"].is_shortcut  # perfect inverse separation
        assert scores["has_https"].roc_auc == pytest.approx(0.0)
        assert not scores["num_dots"].is_shortcut  # random noise

    def test_single_feature_auc_sorted_by_separation(self) -> None:
        dataset = make_dataset(300)
        scores = single_feature_roc_auc(dataset.X, dataset.y, FEATURE_NAMES)

        separations = [s.separation for s in scores]
        assert separations == sorted(separations, reverse=True)

    def test_random_forest_importance_ranks_signal_feature_first(self) -> None:
        dataset = make_dataset(400)
        dataset.X[:, HTTPS] = 0  # leave path_length as the only signal
        model = build_random_forest().fit(dataset.X, dataset.y)

        importances = random_forest_feature_importance(
            model, dataset.X, dataset.y, FEATURE_NAMES, seed=0, n_repeats=2
        )

        assert len(importances) == N_FEATURES
        assert importances[0].feature == "path_length"
        assert sum(i.impurity_importance for i in importances) == pytest.approx(1.0)

    def test_logistic_coefficients_sign(self) -> None:
        dataset = make_dataset(300)
        model = build_logistic_regression().fit(dataset.X, dataset.y)

        coefficients = dict(logistic_regression_coefficients(model, FEATURE_NAMES))
        assert coefficients["path_length"] > 0
        assert coefficients["has_https"] < 0


class TestSchemeBias:
    def test_mask_reads_semicolon_separated_actions(self) -> None:
        dataset = make_dataset(
            4,
            actions=["", "added_default_scheme", "removed_wrapping_quotes;added_default_scheme", "x_added_default_scheme"],
            hosts_per_group=1,
        )

        assert scheme_added_mask(dataset).tolist() == [False, True, True, False]

    def test_summary_counts(self) -> None:
        n = 30
        actions = ["added_default_scheme" if i % 3 and i < 12 else "" for i in range(n)]
        dataset = make_dataset(n, actions=actions, hosts_per_group=1)

        summary = summarize_scheme_bias(dataset, HTTPS)

        flagged = scheme_added_mask(dataset)
        assert summary.flagged_rows == int(flagged.sum())
        assert summary.flagged_legitimate == 0
        assert summary.single_class
        assert summary.https_rate_legitimate == 1.0
        assert summary.phishing_http_rows == int((dataset.y == 1).sum())

    def test_ablation_scores_on_unflagged_test_rows(self) -> None:
        train = make_dataset(300, seed=1, actions=["added_default_scheme" if i % 10 == 1 else "" for i in range(300)])
        test = make_dataset(150, seed=2, actions=["added_default_scheme" if i % 10 == 1 else "" for i in range(150)])
        model = build_random_forest().fit(train.X, train.y)

        ablation = run_scheme_ablation(RANDOM_FOREST, model, build_random_forest, train, test)

        assert ablation.train_flagged_rows == 30
        assert ablation.test_flagged_rows == 15
        assert ablation.with_flagged_on_clean_test.n_samples == 135
        assert ablation.without_flagged_on_clean_test.n_samples == 135
        assert ablation.with_flagged_on_flagged_test is not None
        assert ablation.with_flagged_on_flagged_test.n_samples == 15

    def _ablation(self, f1_with: float, f1_without: float):
        y = np.array([1, 0])

        def result(f1: float):
            base = compute_metrics(y, y, np.array([0.9, 0.1]))
            return base.__class__(**{**base.to_dict(), "f1_phishing": f1})

        return SchemeAblation("m", 1, 1, result(f1_with), result(f1_without), None)

    def _summary(self, flagged_phish: int, flagged_legit: int, phishing_http: int):
        dataset = make_dataset(6, hosts_per_group=1)
        summary = summarize_scheme_bias(dataset, HTTPS)
        return summary.__class__(
            **{
                **{k: v for k, v in summary.to_dict().items() if k != "single_class"},
                "flagged_rows": flagged_phish + flagged_legit,
                "flagged_phishing": flagged_phish,
                "flagged_legitimate": flagged_legit,
                "phishing_http_rows": phishing_http,
                "flagged_share_of_phishing_http": flagged_phish / phishing_http,
            }
        )

    def test_verdict_no_artifact(self) -> None:
        assert assess_scheme_bias(self._summary(0, 0, 10), []).startswith("NO ARTIFACT")

    def test_verdict_low_when_both_classes_flagged(self) -> None:
        assert assess_scheme_bias(self._summary(5, 5, 100), []).startswith("LOW")

    def test_verdict_weak_for_small_single_class_artifact(self) -> None:
        verdict = assess_scheme_bias(self._summary(2, 0, 100), [self._ablation(0.99, 0.99)])
        assert verdict.startswith("WEAK")

    def test_verdict_strong_for_large_share(self) -> None:
        verdict = assess_scheme_bias(self._summary(50, 0, 100), [self._ablation(0.99, 0.99)])
        assert verdict.startswith("STRONG")

    def test_verdict_strong_for_large_metric_shift(self) -> None:
        verdict = assess_scheme_bias(self._summary(2, 0, 100), [self._ablation(0.99, 0.95)])
        assert verdict.startswith("STRONG")
