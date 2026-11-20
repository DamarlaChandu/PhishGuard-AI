"""Tests for shared Public Suffix List domain extraction."""

import socket

import pytest

from app.ml.features.domain import _build_psl_extractor, extract_domain_info
from app.ml.features.extractor import count_subdomains, extract_url_features


@pytest.mark.parametrize(
    ("hostname", "public_suffix", "registrable_domain", "subdomain_count"),
    [
        ("fudan.edu.cn", "edu.cn", "fudan.edu.cn", 0),
        ("www.pku.edu.cn", "edu.cn", "pku.edu.cn", 1),
        ("www.tsinghua.edu.cn", "edu.cn", "tsinghua.edu.cn", 1),
        ("ugm.ac.id", "ac.id", "ugm.ac.id", 0),
        ("www.ui.ac.id", "ac.id", "ui.ac.id", 1),
        ("s.stis.ac.id", "ac.id", "stis.ac.id", 1),
        ("tenant.blogspot.com", "blogspot.com", "tenant.blogspot.com", 0),
        ("a.b.example.co.uk", "co.uk", "example.co.uk", 2),
        ("example.com.", "com", "example.com", 0),
    ],
)
def test_extract_domain_info(
    hostname: str,
    public_suffix: str,
    registrable_domain: str,
    subdomain_count: int,
) -> None:
    info = extract_domain_info(hostname)

    assert info.public_suffix == public_suffix
    assert info.registrable_domain == registrable_domain
    assert info.subdomain_count == subdomain_count


@pytest.mark.parametrize(
    "hostname",
    [
        "192.0.2.1",
        "2001:db8::1",
        "localhost",
        "intranet",
        "unknown.invalidtld",
        "bad..example.com",
        "-bad.example.com",
        "bad-.example.com",
    ],
)
def test_non_registrable_or_malformed_host_has_no_psl_domain(hostname: str) -> None:
    info = extract_domain_info(hostname)

    assert info.public_suffix == ""
    assert info.registrable_domain == ""
    assert info.subdomain_count == 0


@pytest.mark.parametrize(
    ("url", "expected_subdomains"),
    [
        ("https://www.pku.edu.cn/login", 1),
        ("https://www.tsinghua.edu.cn/", 1),
        ("https://s.stis.ac.id/", 1),
        ("https://tenant.blogspot.com/", 0),
        ("a.b.example.co.uk/path", 2),
        ("http://192.0.2.1/", 0),
        ("http://[2001:db8::1]/", 0),
        ("http://localhost/", 0),
    ],
)
def test_feature_subdomain_count_uses_shared_psl_result(
    url: str, expected_subdomains: int
) -> None:
    features = extract_url_features(url)

    assert features.num_subdomains == expected_subdomains


def test_count_subdomains_delegates_to_shared_domain_utility() -> None:
    hostname = "a.b.tenant.blogspot.com"
    info = extract_domain_info(hostname)

    assert count_subdomains(hostname) == info.subdomain_count == 2


def test_extractor_uses_bundled_snapshot_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(f"network access attempted: {_args!r}, {_kwargs!r}")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    monkeypatch.setattr(socket, "gethostbyname", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket.socket, "connect", fail)

    extracted = _build_psl_extractor()("tenant.blogspot.com")

    assert extracted.top_domain_under_public_suffix == "tenant.blogspot.com"
    assert extracted.is_private


def test_training_group_uses_shared_registrable_domain() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from app.ml.training.training_data import registrable_domain

    for hostname in ("www.pku.edu.cn", "tenant.blogspot.com", "a.b.example.co.uk"):
        info = extract_domain_info(hostname)
        assert registrable_domain(hostname) == info.registrable_domain
        assert count_subdomains(hostname) == info.subdomain_count
        assert (
            extract_url_features(f"https://{hostname}/").num_subdomains
            == info.subdomain_count
        )

    assert registrable_domain("192.0.2.1") == "192.0.2.1"
    assert registrable_domain("localhost") == "localhost"
