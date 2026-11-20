"""Safe validation and normalization of raw URL strings.

Only string parsing happens here: no DNS lookups, no HTTP requests, and no
page content is fetched or executed.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from app.ml.features.constants import (
    ALLOWED_SCHEMES,
    DEFAULT_SCHEME,
    MAX_HOSTNAME_LENGTH,
    MAX_LABEL_LENGTH,
    MAX_URL_LENGTH,
)
from app.ml.features.ip_address import parse_ip_hostname

# "scheme://..." using RFC 3986 scheme syntax.
_EXPLICIT_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
# "something:" at the start: either a scheme ("javascript:") or host:port.
_SCHEME_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
# "host:port", optionally followed by a path, query or fragment.
_HOST_PORT_RE = re.compile(r"^[^:/?#@]+:\d+(?:[/?#]|$)")
# One DNS label: Unicode letters, digits, underscore or hyphen.
_LABEL_RE = re.compile(r"^[\w\-]+$")
# Any whitespace or ASCII control character.
_FORBIDDEN_CHARS_RE = re.compile(r"[\s\x00-\x1f\x7f]")


class InvalidURLError(ValueError):
    """Raised when an input cannot be treated as an http(s) URL."""


@dataclass(frozen=True, slots=True)
class NormalizedURL:
    """A validated URL and its parsed components.

    Attributes:
        url: Normalized URL string (scheme and host lowercased, trailing host
            dot and empty port removed).
        scheme: "http" or "https".
        hostname: Lowercased host without brackets, userinfo, port or trailing dot.
        port: Explicit port number, or None when the URL has none.
        path: Path component, as given.
        query: Query string without the leading "?".
        fragment: Fragment without the leading "#".
        has_userinfo: True when the authority contains "user[:pass]@".
        scheme_was_explicit: False when the default scheme was assumed.
    """

    url: str
    scheme: str
    hostname: str
    port: int | None
    path: str
    query: str
    fragment: str
    has_userinfo: bool
    scheme_was_explicit: bool


def normalize_url(raw_url: object) -> NormalizedURL:
    """Validate ``raw_url`` and return its normalized form.

    Raises:
        InvalidURLError: if the input is not a non-empty string, is too long,
            contains whitespace or control characters, uses a scheme other than
            http/https, or has a missing or malformed host or port.
    """
    if not isinstance(raw_url, str):
        raise InvalidURLError(f"URL must be a string, got {type(raw_url).__name__}")

    text = raw_url.strip()
    if not text:
        raise InvalidURLError("URL must not be empty")
    if len(text) > MAX_URL_LENGTH:
        raise InvalidURLError(f"URL exceeds maximum length of {MAX_URL_LENGTH}")
    if _FORBIDDEN_CHARS_RE.search(text):
        raise InvalidURLError("URL must not contain whitespace or control characters")

    text, scheme_was_explicit = _ensure_scheme(text)

    try:
        parts = urlsplit(text)
        port = parts.port  # Raises ValueError for non-numeric or out-of-range ports.
    except ValueError as exc:
        raise InvalidURLError(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError(f"Unsupported URL scheme: {scheme!r}")

    hostname = _validate_hostname(parts.hostname)
    netloc = _build_netloc(parts.netloc, hostname, port)

    normalized = urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))
    return NormalizedURL(
        url=normalized,
        scheme=scheme,
        hostname=hostname,
        port=port,
        path=parts.path,
        query=parts.query,
        fragment=parts.fragment,
        has_userinfo="@" in parts.netloc,
        scheme_was_explicit=scheme_was_explicit,
    )


def _ensure_scheme(text: str) -> tuple[str, bool]:
    """Return the URL with a scheme, and whether one was present originally."""
    if _EXPLICIT_SCHEME_RE.match(text):
        return text, True
    if text.startswith("//"):
        return f"{DEFAULT_SCHEME}:{text}", False
    if _SCHEME_PREFIX_RE.match(text) and not _HOST_PORT_RE.match(text):
        # e.g. "javascript:alert(1)", "mailto:x@y", "data:text/html,..."
        scheme = text.split(":", 1)[0].lower()
        raise InvalidURLError(f"Unsupported URL scheme: {scheme!r}")
    return f"{DEFAULT_SCHEME}://{text}", False


def _validate_hostname(hostname: str | None) -> str:
    """Return a lowercased, validated hostname or raise InvalidURLError."""
    host = (hostname or "").lower().rstrip(".")
    if not host:
        raise InvalidURLError("URL must include a hostname")
    if parse_ip_hostname(host) is not None:
        return host
    if len(host) > MAX_HOSTNAME_LENGTH:
        raise InvalidURLError("Hostname is too long")

    for label in host.split("."):
        if not label:
            raise InvalidURLError("Hostname contains an empty label")
        if len(label) > MAX_LABEL_LENGTH:
            raise InvalidURLError("Hostname label is too long")
        if not _LABEL_RE.match(label):
            raise InvalidURLError(f"Hostname contains invalid characters: {host!r}")
    return host


def _build_netloc(original_netloc: str, hostname: str, port: int | None) -> str:
    """Rebuild the authority from the validated host, keeping userinfo as given.

    An empty port ("example.com:") is dropped, and IPv6 hosts are re-bracketed.
    """
    userinfo, at, _ = original_netloc.rpartition("@")
    host = f"[{hostname}]" if ":" in hostname else hostname
    port_part = f":{port}" if port is not None else ""
    return f"{userinfo}{at}{host}{port_part}"
