"""Analysis of the "added_default_scheme" cleaning action as a label artifact.

Rows whose URL had no scheme were given "http://" during preparation. If
those rows are concentrated in one class, ``has_https`` (and length features)
may encode a data-collection artifact rather than a phishing signal. This
module measures that; it never removes rows from the dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from app.ml.training.dataset_preparation import ACTION_ADDED_DEFAULT_SCHEME
from app.ml.training.evaluation import EvaluationResult, evaluate_classifier
from app.ml.training.training_data import FeatureDataset

# Thresholds for calling the artifact "strong".
SHARE_THRESHOLD = 0.25  # flagged rows as a share of their class's http rows
METRIC_DELTA_THRESHOLD = 0.01  # change in test F1 / ROC-AUC from ablation


def scheme_added_mask(dataset: FeatureDataset) -> np.ndarray:
    """Boolean mask of rows whose cleaning actions include the added scheme."""
    return np.array(
        [ACTION_ADDED_DEFAULT_SCHEME in actions.split(";") for actions in dataset.cleaning_actions],
        dtype=bool,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class SchemeBiasSummary:
    total_rows: int
    flagged_rows: int
    flagged_phishing: int
    flagged_legitimate: int
    phishing_http_rows: int  # phishing rows with has_https == 0
    flagged_share_of_phishing_http: float | None
    phishing_rate_among_http: float | None
    phishing_rate_among_http_excluding_flagged: float | None
    https_rate_phishing: float | None
    https_rate_phishing_excluding_flagged: float | None
    https_rate_legitimate: float | None

    @property
    def single_class(self) -> bool:
        return self.flagged_rows > 0 and (
            self.flagged_phishing == 0 or self.flagged_legitimate == 0
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "single_class": self.single_class}


def summarize_scheme_bias(dataset: FeatureDataset, https_index: int) -> SchemeBiasSummary:
    """Describe how flagged rows relate to the label and the has_https feature."""
    flagged = scheme_added_mask(dataset)
    https = dataset.X[:, https_index] == 1
    phishing = dataset.y == 1
    http = ~https

    return SchemeBiasSummary(
        total_rows=len(dataset),
        flagged_rows=int(flagged.sum()),
        flagged_phishing=int((flagged & phishing).sum()),
        flagged_legitimate=int((flagged & ~phishing).sum()),
        phishing_http_rows=int((phishing & http).sum()),
        flagged_share_of_phishing_http=_rate(
            int((flagged & phishing).sum()), int((phishing & http).sum())
        ),
        phishing_rate_among_http=_rate(int((phishing & http).sum()), int(http.sum())),
        phishing_rate_among_http_excluding_flagged=_rate(
            int((phishing & http & ~flagged).sum()), int((http & ~flagged).sum())
        ),
        https_rate_phishing=_rate(int((phishing & https).sum()), int(phishing.sum())),
        https_rate_phishing_excluding_flagged=_rate(
            int((phishing & https & ~flagged).sum()), int((phishing & ~flagged).sum())
        ),
        https_rate_legitimate=_rate(int((~phishing & https).sum()), int((~phishing).sum())),
    )


@dataclass(frozen=True)
class SchemeAblation:
    """Effect of the flagged rows on a model, measured on unflagged test rows."""

    model_name: str
    train_flagged_rows: int
    test_flagged_rows: int
    with_flagged_on_clean_test: EvaluationResult
    without_flagged_on_clean_test: EvaluationResult
    with_flagged_on_flagged_test: EvaluationResult | None

    @property
    def f1_delta(self) -> float:
        return self.without_flagged_on_clean_test.f1_phishing - self.with_flagged_on_clean_test.f1_phishing

    @property
    def roc_auc_delta(self) -> float | None:
        a = self.without_flagged_on_clean_test.roc_auc
        b = self.with_flagged_on_clean_test.roc_auc
        return None if a is None or b is None else a - b


def run_scheme_ablation(
    model_name: str,
    fitted_model,
    build_model: Callable[[], object],
    train: FeatureDataset,
    test: FeatureDataset,
) -> SchemeAblation:
    """Compare the fitted model with one retrained without flagged train rows.

    Both are scored on test rows that were not flagged, so the comparison is
    not affected by the flagged rows themselves. The original model is also
    scored on the flagged test rows to show how it treats them.
    """
    train_flagged = scheme_added_mask(train)
    test_flagged = scheme_added_mask(test)
    clean_test = test.subset(np.flatnonzero(~test_flagged))

    ablated = build_model()
    clean_train = train.subset(np.flatnonzero(~train_flagged))
    ablated.fit(clean_train.X, clean_train.y)

    flagged_result = None
    if test_flagged.any():
        flagged_test = test.subset(np.flatnonzero(test_flagged))
        flagged_result = evaluate_classifier(fitted_model, flagged_test.X, flagged_test.y)

    return SchemeAblation(
        model_name=model_name,
        train_flagged_rows=int(train_flagged.sum()),
        test_flagged_rows=int(test_flagged.sum()),
        with_flagged_on_clean_test=evaluate_classifier(fitted_model, clean_test.X, clean_test.y),
        without_flagged_on_clean_test=evaluate_classifier(ablated, clean_test.X, clean_test.y),
        with_flagged_on_flagged_test=flagged_result,
    )


def assess_scheme_bias(summary: SchemeBiasSummary, ablations: list[SchemeAblation]) -> str:
    """Plain-language verdict combining the data summary and the ablations."""
    if summary.flagged_rows == 0:
        return "NO ARTIFACT: no rows had a default scheme added."

    share = summary.flagged_share_of_phishing_http or 0.0
    deltas = [abs(a.f1_delta) for a in ablations]
    deltas += [abs(a.roc_auc_delta) for a in ablations if a.roc_auc_delta is not None]
    max_delta = max(deltas, default=0.0)

    if not summary.single_class:
        return (
            "LOW: flagged rows occur in both classes, so the added scheme is not "
            "a class-specific artifact."
        )
    if share >= SHARE_THRESHOLD or max_delta >= METRIC_DELTA_THRESHOLD:
        return (
            f"STRONG: flagged rows are all one class, make up {share:.1%} of "
            f"phishing http rows, and removing them shifts test metrics by up to "
            f"{max_delta:.4f}. Treat has_https with caution."
        )
    return (
        f"WEAK: flagged rows are all one class (a real artifact), but they are only "
        f"{share:.1%} of phishing http rows and removing them shifts test F1/ROC-AUC "
        f"by at most {max_delta:.4f} (< {METRIC_DELTA_THRESHOLD}). Kept in the "
        f"dataset; monitor it."
    )
