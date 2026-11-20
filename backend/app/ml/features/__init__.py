"""URL feature extraction for PhishGuard AI.

Usage::

    from app.ml.features import extract_url_features

    features = extract_url_features("https://example.com/login?id=1")
    features.to_dict()    # {"url_length": 30, "hostname_length": 11, ...}
    features.to_vector()  # values in FEATURE_NAMES order
"""

from app.ml.features.extractor import (
    FEATURE_NAMES,
    URLFeatures,
    extract_features_from_normalized,
    extract_url_features,
)
from app.ml.features.domain import DomainInfo, domain_group_key, extract_domain_info
from app.ml.features.normalization import InvalidURLError, NormalizedURL, normalize_url

__all__ = [
    "DomainInfo",
    "FEATURE_NAMES",
    "InvalidURLError",
    "NormalizedURL",
    "URLFeatures",
    "domain_group_key",
    "extract_domain_info",
    "extract_features_from_normalized",
    "extract_url_features",
    "normalize_url",
]
