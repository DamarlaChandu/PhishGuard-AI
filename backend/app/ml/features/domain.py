"""Offline Public Suffix List based hostname decomposition."""

import re
from dataclasses import dataclass

from tldextract import TLDExtract

from app.ml.features.ip_address import parse_ip_hostname

_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


@dataclass(frozen=True, slots=True)
class DomainInfo:
    """PSL decomposition of a hostname.

    Non-domain hosts and malformed hostnames have no public suffix or
    registrable domain and report zero subdomains.
    """

    public_suffix: str
    registrable_domain: str
    subdomain_count: int


_EMPTY_DOMAIN_INFO = DomainInfo(
    public_suffix="",
    registrable_domain="",
    subdomain_count=0,
)

# tldextract 5.4.0 bundles a PSL snapshot. An empty URL list forces that
# snapshot to be used; disabling the cache avoids filesystem-dependent state.
# Private suffixes are included so hosted tenants (for example Blogspot blogs)
# each have their own registrable domain.
def _build_psl_extractor() -> TLDExtract:
    return TLDExtract(
        cache_dir=None,
        suffix_list_urls=(),
        fallback_to_snapshot=True,
        include_psl_private_domains=True,
    )


_PSL_EXTRACTOR = _build_psl_extractor()


def extract_domain_info(hostname: str) -> DomainInfo:
    """Return the PSL suffix, eTLD+1, and subdomain count for a hostname.

    The input is a hostname, not a full URL. Callers handling URLs should parse
    them first. IP addresses, single-label hosts, unknown suffixes, and invalid
    hostnames are not reported as registrable domains.
    """
    if not isinstance(hostname, str):
        return _EMPTY_DOMAIN_INFO

    host = hostname.strip().lower()
    if host.endswith("."):
        host = host[:-1]
    if not host or len(host) > 253:
        return _EMPTY_DOMAIN_INFO
    if parse_ip_hostname(host) is not None:
        return _EMPTY_DOMAIN_INFO

    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return _EMPTY_DOMAIN_INFO
    if len(ascii_host) > 253:
        return _EMPTY_DOMAIN_INFO

    labels = ascii_host.split(".")
    if any(not _HOST_LABEL_RE.fullmatch(label) for label in labels):
        return _EMPTY_DOMAIN_INFO

    extracted = _PSL_EXTRACTOR(ascii_host)
    public_suffix = extracted.suffix.lower()
    registrable_domain = extracted.top_domain_under_public_suffix.lower()
    if not public_suffix or not registrable_domain:
        return DomainInfo(
            public_suffix=public_suffix,
            registrable_domain="",
            subdomain_count=0,
        )

    return DomainInfo(
        public_suffix=public_suffix,
        registrable_domain=registrable_domain,
        subdomain_count=len(extracted.subdomain.split(".")) if extracted.subdomain else 0,
    )
