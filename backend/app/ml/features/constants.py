"""Static lookup tables and limits used by URL normalization and feature extraction.

These lists are heuristic indicators, not verdicts. Keep them small, documented
and lowercase; matching code lowercases its input before comparing.
"""

from typing import Final

# Inputs longer than this are rejected to bound memory and CPU per request.
MAX_URL_LENGTH: Final[int] = 8192

# DNS limits (RFC 1035).
MAX_HOSTNAME_LENGTH: Final[int] = 253
MAX_LABEL_LENGTH: Final[int] = 63

ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# Scheme assumed when the input has none (e.g. "example.com/login").
DEFAULT_SCHEME: Final[str] = "http"

# Words commonly used in credential-harvesting and account-takeover lures.
SUSPICIOUS_KEYWORDS: Final[tuple[str, ...]] = (
    "login",
    "log-in",
    "signin",
    "sign-in",
    "logon",
    "verify",
    "verification",
    "validate",
    "account",
    "update",
    "secure",
    "security",
    "banking",
    "confirm",
    "password",
    "passwd",
    "credential",
    "authenticate",
    "wallet",
    "suspend",
    "unlock",
    "restore",
    "billing",
    "invoice",
    "payment",
    "webscr",
    "ebayisapi",
)

# Well-known public URL-shortening services. Subdomains also match.
URL_SHORTENER_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "bit.ly",
        "bitly.com",
        "goo.gl",
        "tinyurl.com",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "bit.do",
        "cutt.ly",
        "rebrand.ly",
        "shorturl.at",
        "tiny.cc",
        "rb.gy",
        "t.ly",
        "v.gd",
        "lnkd.in",
        "trib.al",
        "s.id",
        "shorte.st",
        "x.co",
        "qr.ae",
        "bl.ink",
        "soo.gd",
        "clck.ru",
        "u.to",
        "short.io",
        "tr.im",
        "mcaf.ee",
    }
)

# Characters already counted by dedicated features (dots, hyphens) or that
# structure every URL ("/" and ":"); excluded from the special-character count.
NON_SPECIAL_PUNCTUATION: Final[frozenset[str]] = frozenset({".", "-", "/", ":"})
