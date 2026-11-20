"""Tests for numerical URL feature extraction."""

import socket

import pytest

from app.ml.features import (
    FEATURE_NAMES,
    InvalidURLError,
    URLFeatures,
    extract_url_features,
)
from app.ml.features.ip_address import parse_ip_hostname

EXPECTED_FEATURE_NAMES = (
    "url_length",
    "hostname_length",
    "num_dots",
    "num_hyphens",
    "num_subdomains",
    "num_digits",
    "num_special_chars",
    "has_https",
    "has_ip_address",
    "has_suspicious_keywords",
    "has_at_symbol",
    "has_port",
    "path_length",
    "num_parameters",
    "is_shortened_url",
)


class TestOutputStructure:
    def test_feature_names_are_stable_and_ordered(self) -> None:
        assert FEATURE_NAMES == EXPECTED_FEATURE_NAMES

    def test_dict_and_vector_are_consistent(self) -> None:
        features = extract_url_features("https://example.com/a?b=1")

        assert isinstance(features, URLFeatures)
        assert tuple(features.to_dict()) == FEATURE_NAMES
        assert features.to_vector() == list(features.to_dict().values())

    def test_all_values_are_non_negative_integers(self) -> None:
        features = extract_url_features("http://user@1.2.3.4:81/a-b/login?x=1&y=2")

        for value in features.to_vector():
            assert type(value) is int
            assert value >= 0

    def test_features_are_deterministic(self) -> None:
        url = "https://secure-login.example.com/verify?id=123"
        assert extract_url_features(url) == extract_url_features(url)

    def test_features_are_immutable(self) -> None:
        features = extract_url_features("https://example.com")
        with pytest.raises(AttributeError):
            features.url_length = 0  # type: ignore[misc]


class TestNormalURLs:
    def test_simple_https_url(self) -> None:
        features = extract_url_features("https://www.example.com/")

        assert features.to_dict() == {
            "url_length": 24,
            "hostname_length": 15,
            "num_dots": 2,
            "num_hyphens": 0,
            "num_subdomains": 1,
            "num_digits": 0,
            "num_special_chars": 0,
            "has_https": 1,
            "has_ip_address": 0,
            "has_suspicious_keywords": 0,
            "has_at_symbol": 0,
            "has_port": 0,
            "path_length": 1,
            "num_parameters": 0,
            "is_shortened_url": 0,
        }

    def test_url_with_path_and_query(self) -> None:
        features = extract_url_features("https://docs.python.org/3/library/urllib.html?a=1&b=2")

        assert features.url_length == 53
        assert features.hostname_length == 15
        assert features.num_dots == 3
        assert features.num_subdomains == 1
        assert features.num_digits == 3
        # "?", "=", "&", "=" are special; ".", "/", ":" are not.
        assert features.num_special_chars == 4
        assert features.path_length == 22
        assert features.num_parameters == 2

    def test_bare_domain_has_no_subdomains(self) -> None:
        features = extract_url_features("https://example.com")

        assert features.num_subdomains == 0
        assert features.path_length == 0

    def test_scheme_less_input_counts_as_http(self) -> None:
        features = extract_url_features("example.com")

        assert features.has_https == 0
        assert features.url_length == len("http://example.com")

    def test_case_differences_in_host_do_not_change_features(self) -> None:
        assert extract_url_features("https://EXAMPLE.com/a") == extract_url_features(
            "https://example.com/a"
        )


class TestSuspiciousURLs:
    def test_typical_phishing_style_url(self) -> None:
        url = "http://secure-paypal.account-verify.example-login.com/signin/update.php?user=a&session=1"
        features = extract_url_features(url)

        assert features.has_https == 0
        assert features.num_hyphens == 3
        assert features.num_subdomains == 2
        assert features.has_suspicious_keywords == 1
        assert features.num_parameters == 2

    def test_at_symbol_in_authority(self) -> None:
        features = extract_url_features("http://www.paypal.com@evil.example/")

        assert features.has_at_symbol == 1
        assert features.hostname_length == len("evil.example")

    def test_at_symbol_in_path(self) -> None:
        assert extract_url_features("https://example.com/@user").has_at_symbol == 1

    def test_explicit_port(self) -> None:
        features = extract_url_features("http://example.com:8080/")

        assert features.has_port == 1
        assert features.num_digits == 4

    def test_default_port_written_explicitly_still_counts(self) -> None:
        assert extract_url_features("https://example.com:443/").has_port == 1

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/LOGIN",
            "https://login.example.com/",
            "https://example.com/?next=verify",
            "https://example.com/#password",
            "https://example.com/%6Cogin",  # percent-encoded "login"
        ],
    )
    def test_suspicious_keyword_detected_anywhere_after_scheme(self, url: str) -> None:
        assert extract_url_features(url).has_suspicious_keywords == 1

    def test_https_scheme_is_not_a_keyword_match(self) -> None:
        assert extract_url_features("https://example.com/").has_suspicious_keywords == 0

    @pytest.mark.parametrize(
        "url",
        ["https://bit.ly/3xYz", "http://tinyurl.com/abc", "https://www.bit.ly/x", "t.co/abc"],
    )
    def test_url_shorteners_detected(self, url: str) -> None:
        assert extract_url_features(url).is_shortened_url == 1

    @pytest.mark.parametrize("url", ["https://notbit.ly/", "https://bit.ly.example.com/"])
    def test_lookalike_shortener_domains_not_flagged(self, url: str) -> None:
        assert extract_url_features(url).is_shortened_url == 0

    def test_special_characters_counted(self) -> None:
        # "~", "_", "%", "2", "0" -> special chars are "~", "_", "%"
        features = extract_url_features("https://example.com/~user_name%20x")
        assert features.num_special_chars == 3


class TestIPBasedURLs:
    def test_ipv4_host(self) -> None:
        features = extract_url_features("http://192.168.1.10/login")

        assert features.has_ip_address == 1
        assert features.num_subdomains == 0
        assert features.hostname_length == 12
        assert features.num_dots == 3

    def test_ipv4_with_port(self) -> None:
        features = extract_url_features("http://10.0.0.1:8080/admin")

        assert features.has_ip_address == 1
        assert features.has_port == 1

    def test_ipv6_host(self) -> None:
        features = extract_url_features("http://[2001:db8::1]/")

        assert features.has_ip_address == 1
        assert features.hostname_length == len("2001:db8::1")
        assert features.num_subdomains == 0

    @pytest.mark.parametrize(
        "host",
        ["3232235777", "0xC0A80001", "0300.0250.0.1", "0xc0.0xa8.0x00.0x01", "192.168.257"],
    )
    def test_obfuscated_ipv4_hosts(self, host: str) -> None:
        assert extract_url_features(f"http://{host}/").has_ip_address == 1

    def test_domain_starting_with_digits_is_not_an_ip(self) -> None:
        assert extract_url_features("https://123.example.com/").has_ip_address == 0

    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("127.0.0.1", "127.0.0.1"),
            ("127.1", "127.0.0.1"),
            ("3232235777", "192.168.1.1"),
            ("0xC0A80001", "192.168.0.1"),
            ("0300.0250.0.1", "192.168.0.1"),
            ("[::1]", "::1"),
        ],
    )
    def test_parse_ip_hostname_values(self, host: str, expected: str) -> None:
        assert str(parse_ip_hostname(host)) == expected

    @pytest.mark.parametrize(
        "host",
        ["example.com", "256.1.1.1", "4294967296", "1.2.3.4.5", "09.1.1.1", "1..2", "0xZZ"],
    )
    def test_parse_ip_hostname_rejects_non_ips(self, host: str) -> None:
        assert parse_ip_hostname(host) is None


class TestEdgeCases:
    def test_multi_label_public_suffix(self) -> None:
        assert extract_url_features("https://shop.example.co.uk/").num_subdomains == 1
        assert extract_url_features("https://example.co.uk/").num_subdomains == 0

    def test_deeply_nested_subdomains(self) -> None:
        assert extract_url_features("http://a.b.c.d.example.com/").num_subdomains == 4

    def test_single_label_host(self) -> None:
        features = extract_url_features("http://localhost/")

        assert features.num_subdomains == 0
        assert features.has_ip_address == 0

    def test_empty_and_repeated_query_separators_ignored(self) -> None:
        assert extract_url_features("https://example.com/?a=1&&b=2&").num_parameters == 2
        assert extract_url_features("https://example.com/?").num_parameters == 0

    def test_query_parameter_without_value_counts(self) -> None:
        assert extract_url_features("https://example.com/?flag").num_parameters == 1

    def test_fragment_counts_toward_url_length_not_path(self) -> None:
        features = extract_url_features("https://example.com/a#section")

        assert features.path_length == 2
        assert features.url_length == len("https://example.com/a#section")

    def test_long_url_within_limit(self) -> None:
        url = "https://example.com/" + "a" * 5000
        assert extract_url_features(url).path_length == 5001

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(InvalidURLError):
            extract_url_features("javascript:alert(document.cookie)")

    def test_extraction_makes_no_network_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("network access attempted")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        monkeypatch.setattr(socket, "gethostbyname", fail)
        monkeypatch.setattr(socket, "create_connection", fail)
        monkeypatch.setattr(socket.socket, "connect", fail)

        extract_url_features("https://login.example.com:8443/verify?a=1")
        extract_url_features("http://0xC0A80001/")
