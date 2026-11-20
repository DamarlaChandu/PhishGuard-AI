"""Baseline classifiers for URL phishing detection."""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RANDOM_FOREST = "random_forest"
LOGISTIC_REGRESSION = "logistic_regression"

DEFAULT_SEED = 42


def build_random_forest(seed: int = DEFAULT_SEED) -> RandomForestClassifier:
    """Random Forest; tree models need no feature scaling.

    ``class_weight="balanced"`` offsets the roughly 63/37 class imbalance.
    """
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def build_logistic_regression(seed: int = DEFAULT_SEED) -> Pipeline:
    """Logistic Regression with log1p + standardization.

    The count features (lengths, digits, dots...) are non-negative and heavily
    right-skewed, so ``log1p`` compresses long tails before ``StandardScaler``
    centres and scales every feature. Both steps are fitted on training data
    only, as part of the pipeline.
    """
    return Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
        ]
    )


def build_baseline_models(seed: int = DEFAULT_SEED) -> dict[str, ClassifierMixin]:
    return {
        RANDOM_FOREST: build_random_forest(seed),
        LOGISTIC_REGRESSION: build_logistic_regression(seed),
    }
