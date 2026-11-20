"""Numerical URL feature extraction for machine-learning models.

Every feature is computed from the URL string alone. Nothing is fetched,
resolved or executed.
"""

from dataclasses import asdict, dataclass, fields
from urllib.parse import unquote

from app.ml.features.constants import (
    NON_SPECIAL_PUNCTUATION,
    SUSPICIOUS_KEYWORDS,
    URL_SHORTENER_DOMAINS,
)
from app.ml.features.domain import extract_domain_info
from app.ml.features.ip_address import parse_ip_hostname
from app.ml.features.normalization import NormalizedURL, normalize_url


@dataclass(frozen=True, slots=True)
class URLFeatures:
    """Fixed-order numerical feature set for one URL.

    Field order defines the order of :meth:`to_vector` and :data:`FEATURE_NAMES`;
    append new features at the end so existing model inputs stay aligned.
    Binary indicators are 0 or 1.
    """

    url_length: int
    hostname_length: int
    num_dots: int
    num_hyphens: int
    num_subdomains: int
    num_digits: int
    num_special_chars: int
    has_https: int
    has_ip_address: int
    has_suspicious_keywords: int
    has_at_symbol: int
    has_port: int
    path_length: int
    num_parameters: int
    is_shortened_url: int

    def to_dict(self) -> dict[str, int]:
        """Return features as a name-to-value mapping, in FEATURE_NAMES order."""
        return asdict(self)

    def to_vector(self) -> list[int]:
        """Return feature values in FEATURE_NAMES order."""
        return [getattr(self, name) for name in FEATURE_NAMES]


FEATURE_NAMES: tuple[str, ...] = tuple(field.name for field in fields(URLFeatures))


def extract_url_features(raw_url: object) -> URLFeatures:
    """Validate, normalize and extract features from a raw URL.

    Raises:
        InvalidURLError: if the input is not a valid http(s) URL.
    """
    return extract_features_from_normalized(normalize_url(raw_url))


def extract_features_from_normalized(parsed: NormalizedURL) -> URLFeatures:
    """Extract features from an already-normalized URL."""
    url = parsed.url
    is_ip = parse_ip_hostname(parsed.hostname) is not None
    domain_info = extract_domain_info(parsed.hostname)

    return URLFeatures(
        url_length=len(url),
        hostname_length=len(parsed.hostname),
        num_dots=url.count("."),
        num_hyphens=url.count("-"),
        num_subdomains=domain_info.subdomain_count,
        num_digits=sum(ch.isdigit() for ch in url),
        num_special_chars=count_special_chars(url),
        has_https=int(parsed.scheme == "https"),
        has_ip_address=int(is_ip),
        has_suspicious_keywords=int(contains_suspicious_keyword(parsed)),
        has_at_symbol=int("@" in url),
        has_port=int(parsed.port is not None),
        path_length=len(parsed.path),
        num_parameters=count_query_parameters(parsed.query),
        is_shortened_url=int(is_shortener_host(parsed.hostname)),
    )


def count_subdomains(hostname: str) -> int:
    """Return the PSL-derived subdomain count for a hostname."""
    return extract_domain_info(hostname).subdomain_count


def count_special_chars(url: str) -> int:
    """Count non-alphanumeric characters, excluding ``.``, ``-``, ``/`` and ``:``."""
    return sum(
        1 for ch in url if not ch.isalnum() and ch not in NON_SPECIAL_PUNCTUATION
    )


def contains_suspicious_keyword(parsed: NormalizedURL) -> bool:
    """Return True if a lure keyword appears in the host, path, query or fragment.

    The scheme and userinfo are excluded, and percent-encoding is decoded so
    ``%6Cogin`` matches ``login``.
    """
    searchable = unquote(
        f"{parsed.hostname}{parsed.path}?{parsed.query}#{parsed.fragment}"
    ).lower()
    return any(keyword in searchable for keyword in SUSPICIOUS_KEYWORDS)


def count_query_parameters(query: str) -> int:
    """Count non-empty ``&``-separated segments of the query string."""
    return sum(1 for segment in query.split("&") if segment)


def is_shortener_host(hostname: str) -> bool:
    """Return True if the host is, or is a subdomain of, a known URL shortener."""
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in URL_SHORTENER_DOMAINS
    )
