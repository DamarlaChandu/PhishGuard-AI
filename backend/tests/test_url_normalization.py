"""Tests for URL validation and normalization."""

import pytest

from app.ml.features import InvalidURLError, normalize_url
from app.ml.features.constants import MAX_URL_LENGTH


class TestValidURLs:
    def test_keeps_explicit_https_scheme(self) -> None:
        result = normalize_url("https://example.com/path?q=1#top")

        assert result.url == "https://example.com/path?q=1#top"
        assert result.scheme == "https"
        assert result.hostname == "example.com"
        assert result.port is None
        assert result.path == "/path"
        assert result.query == "q=1"
        assert result.fragment == "top"
        assert result.scheme_was_explicit is True

    def test_lowercases_scheme_and_host_but_not_path(self) -> None:
        result = normalize_url("HTTPS://WWW.Example.COM/Login/Page")

        assert result.url == "https://www.example.com/Login/Page"
        assert result.hostname == "www.example.com"

    def test_strips_surrounding_whitespace(self) -> None:
        assert normalize_url("  https://example.com  ").url == "https://example.com"

    def test_adds_default_scheme_when_missing(self) -> None:
        result = normalize_url("example.com/login")

        assert result.url == "http://example.com/login"
        assert result.scheme == "http"
        assert result.scheme_was_explicit is False

    def test_adds_default_scheme_to_protocol_relative_url(self) -> None:
        assert normalize_url("//example.com/a").url == "http://example.com/a"

    def test_host_and_port_without_scheme(self) -> None:
        result = normalize_url("localhost:8080/admin")

        assert result.url == "http://localhost:8080/admin"
        assert result.port == 8080

    def test_removes_trailing_dot_from_hostname(self) -> None:
        result = normalize_url("https://example.com./a")

        assert result.hostname == "example.com"
        assert result.url == "https://example.com/a"

    def test_drops_empty_port(self) -> None:
        assert normalize_url("http://example.com:/a").url == "http://example.com/a"

    def test_preserves_userinfo(self) -> None:
        result = normalize_url("http://User@Example.com/")

        assert result.url == "http://User@example.com/"
        assert result.hostname == "example.com"
        assert result.has_userinfo is True

    def test_ipv6_host_is_rebracketed(self) -> None:
        result = normalize_url("http://[2001:DB8::1]:8443/")

        assert result.hostname == "2001:db8::1"
        assert result.url == "http://[2001:db8::1]:8443/"
        assert result.port == 8443

    def test_accepts_internationalized_hostname(self) -> None:
        assert normalize_url("https://bücher.de/").hostname == "bücher.de"


class TestInvalidURLs:
    @pytest.mark.parametrize("value", [None, 42, 3.5, b"https://example.com", ["x"]])
    def test_rejects_non_string_input(self, value: object) -> None:
        with pytest.raises(InvalidURLError, match="must be a string"):
            normalize_url(value)

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_rejects_empty_input(self, value: str) -> None:
        with pytest.raises(InvalidURLError, match="must not be empty"):
            normalize_url(value)

    @pytest.mark.parametrize(
        "value",
        [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "mailto:user@example.com",
            "ftp://example.com/file",
            "file:///etc/passwd",
            "http:example.com",
        ],
    )
    def test_rejects_unsupported_schemes(self, value: str) -> None:
        with pytest.raises(InvalidURLError, match="Unsupported URL scheme"):
            normalize_url(value)

    @pytest.mark.parametrize(
        "value",
        ["https://exa mple.com", "https://example.com/a b", "https://example.com/\x00"],
    )
    def test_rejects_whitespace_and_control_characters(self, value: str) -> None:
        with pytest.raises(InvalidURLError, match="whitespace or control"):
            normalize_url(value)

    @pytest.mark.parametrize("value", ["http://", "https:///path", "http://:8080/"])
    def test_rejects_missing_hostname(self, value: str) -> None:
        with pytest.raises(InvalidURLError, match="hostname"):
            normalize_url(value)

    @pytest.mark.parametrize(
        "value", ["http://example.com:99999/", "http://example.com:abc/"]
    )
    def test_rejects_invalid_port(self, value: str) -> None:
        with pytest.raises(InvalidURLError, match="Malformed URL"):
            normalize_url(value)

    def test_rejects_empty_hostname_label(self) -> None:
        with pytest.raises(InvalidURLError, match="empty label"):
            normalize_url("http://example..com/")

    def test_rejects_invalid_hostname_characters(self) -> None:
        with pytest.raises(InvalidURLError, match="invalid characters"):
            normalize_url("http://exa$mple.com/")

    def test_rejects_overlong_label(self) -> None:
        with pytest.raises(InvalidURLError, match="label is too long"):
            normalize_url(f"http://{'a' * 64}.com/")

    def test_rejects_overlong_hostname(self) -> None:
        host = ".".join(["a" * 60] * 5)
        with pytest.raises(InvalidURLError, match="Hostname is too long"):
            normalize_url(f"http://{host}/")

    def test_rejects_overlong_url(self) -> None:
        with pytest.raises(InvalidURLError, match="maximum length"):
            normalize_url("https://example.com/" + "a" * MAX_URL_LENGTH)

    def test_rejects_malformed_ipv6(self) -> None:
        with pytest.raises(InvalidURLError):
            normalize_url("http://[::1/")

    def test_invalid_url_error_is_a_value_error(self) -> None:
        assert issubclass(InvalidURLError, ValueError)
