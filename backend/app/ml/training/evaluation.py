"""Evaluation metrics and feature-importance helpers.

Positive class is phishing (label 1) throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

PHISHING = 1
LEGITIMATE = 0
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluationResult:
    n_samples: int
    n_phishing: int
    n_legitimate: int
    threshold: float
    accuracy: float
    precision_phishing: float
    recall_phishing: float
    f1_phishing: float
    precision_legitimate: float
    recall_legitimate: float
    f1_legitimate: float
    f1_macro: float
    roc_auc: float | None  # None when only one class is present
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def confusion_matrix(self) -> list[list[int]]:
        """[[TN, FP], [FN, TP]] with rows = actual, columns = predicted."""
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]


def phishing_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Probability of the phishing class for each row."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(PHISHING)]


def evaluate_classifier(
    model, X: np.ndarray, y: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> EvaluationResult:
    """Compute classification metrics for a fitted probabilistic classifier."""
    if len(y) == 0:
        raise ValueError("cannot evaluate on an empty set")
    probabilities = phishing_probabilities(model, X)
    predictions = (probabilities >= threshold).astype(np.int64)
    return compute_metrics(y, predictions, probabilities, threshold)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phishing_scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> EvaluationResult:
    labels = [LEGITIMATE, PHISHING]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    roc_auc = (
        float(roc_auc_score(y_true, phishing_scores)) if len(np.unique(y_true)) == 2 else None
    )
    return EvaluationResult(
        n_samples=int(len(y_true)),
        n_phishing=int(np.sum(y_true == PHISHING)),
        n_legitimate=int(np.sum(y_true == LEGITIMATE)),
        threshold=threshold,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_phishing=float(precision[1]),
        recall_phishing=float(recall[1]),
        f1_phishing=float(f1[1]),
        precision_legitimate=float(precision[0]),
        recall_legitimate=float(recall[0]),
        f1_legitimate=float(f1[0]),
        f1_macro=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        roc_auc=roc_auc,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


SHORTCUT_SEPARATION_THRESHOLD = 0.95


@dataclass(frozen=True)
class SingleFeatureScore:
    """How well one raw feature alone ranks phishing above legitimate.

    ``separation`` is max(AUC, 1 - AUC): 0.5 means no signal, 1.0 means the
    feature alone perfectly separates the classes in either direction.
    """

    feature: str
    roc_auc: float

    @property
    def separation(self) -> float:
        return max(self.roc_auc, 1.0 - self.roc_auc)

    @property
    def is_shortcut(self) -> bool:
        return self.separation >= SHORTCUT_SEPARATION_THRESHOLD


def single_feature_roc_auc(
    X: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...]
) -> list[SingleFeatureScore]:
    """ROC-AUC of each feature used directly as a score; sorted by separation.

    A feature that nearly separates the classes on its own often indicates a
    dataset-collection artifact rather than a robust phishing signal.
    """
    if len(np.unique(y)) < 2:
        raise ValueError("single-feature ROC-AUC needs both classes")
    scores = [
        SingleFeatureScore(name, float(roc_auc_score(y, X[:, i])))
        for i, name in enumerate(feature_names)
    ]
    return sorted(scores, key=lambda s: s.separation, reverse=True)


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    impurity_importance: float
    permutation_mean: float
    permutation_std: float


def random_forest_feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    seed: int,
    n_repeats: int = 5,
) -> list[FeatureImportance]:
    """Impurity importance plus permutation importance (drop in ROC-AUC).

    Impurity importance comes from training and favours high-cardinality
    features; permutation importance is measured on the given (test) data and
    is the more reliable of the two. Sorted by permutation importance.
    """
    impurity = model.feature_importances_
    permutation = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    items = [
        FeatureImportance(
            feature=name,
            impurity_importance=float(impurity[i]),
            permutation_mean=float(permutation.importances_mean[i]),
            permutation_std=float(permutation.importances_std[i]),
        )
        for i, name in enumerate(feature_names)
    ]
    return sorted(items, key=lambda item: item.permutation_mean, reverse=True)


def logistic_regression_coefficients(
    pipeline: Pipeline, feature_names: tuple[str, ...]
) -> list[tuple[str, float]]:
    """Standardized coefficients, sorted by magnitude (positive -> phishing)."""
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    pairs = [(name, float(coefficients[i])) for i, name in enumerate(feature_names)]
    return sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)
