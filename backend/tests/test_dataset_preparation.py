"""Tests for dataset preparation: labels, cleaning, deduplication and conflicts."""

import csv
from pathlib import Path

import pytest

from app.ml.features import FEATURE_NAMES, InvalidURLError, extract_url_features
from app.ml.training.dataset_preparation import (
    ACTION_ADDED_DEFAULT_SCHEME,
    ACTION_NORMALIZED_BY_BACKEND,
    ACTION_REMOVED_WRAPPING_QUOTES,
    ACTION_TRIMMED_WHITESPACE,
    ACTION_UNESCAPED_PERCENT,
    EXCLUDED_CONFLICTING_LABELS,
    EXCLUDED_EMPTY_URL,
    EXCLUDED_INVALID_URL,
    EXCLUDED_MISSING_LABEL,
    EXCLUDED_UNSUPPORTED_LABEL,
    LABEL_LEGITIMATE,
    LABEL_PHISHING,
    OUTPUT_COLUMNS,
    SOURCE_LEGITPHISH,
    SOURCE_OPENPHISH,
    RawRecord,
    clean_url,
    convert_legitphish_label,
    load_legitphish,
    load_openphish,
    prepare_records,
    validate_dataset_url,
    write_dataset,
)

TIMESTAMP = "2026-01-01T00:00:00Z"


def record(
    url: str,
    label: int | None = LABEL_PHISHING,
    source: str = SOURCE_LEGITPHISH,
    raw_label: str | None = None,
    collected_at: str = TIMESTAMP,
) -> RawRecord:
    if raw_label is None:
        raw_label = "" if label is None else str(label)
    return RawRecord(source, url, raw_label, label, collected_at, "test")


class TestLabelConversion:
    def test_legitphish_phishing_becomes_project_phishing(self) -> None:
        assert convert_legitphish_label("0") == LABEL_PHISHING == 1

    def test_legitphish_legitimate_becomes_project_legitimate(self) -> None:
        assert convert_legitphish_label("1") == LABEL_LEGITIMATE == 0

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert convert_legitphish_label(" 0 ") == LABEL_PHISHING

    @pytest.mark.parametrize("raw", [None, "", "  ", "2", "-1", "0.0", "1.0", "phishing", "yes"])
    def test_missing_or_unsupported_labels_return_none(self, raw: str | None) -> None:
        assert convert_legitphish_label(raw) is None

    def test_openphish_loader_labels_everything_phishing(self, tmp_path: Path) -> None:
        feed = tmp_path / "feed.txt"
        feed.write_text("https://a.example/\n\nhttp://b.example/x\n", encoding="utf-8")

        records = list(load_openphish(feed, TIMESTAMP))

        assert [r.raw_url for r in records] == ["https://a.example/", "http://b.example/x"]
        assert {r.label for r in records} == {LABEL_PHISHING}
        assert {r.source for r in records} == {SOURCE_OPENPHISH}
        assert records[1].location == "feed.txt:3"

    def test_legitphish_loader_converts_labels_and_ignores_feature_columns(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "lp.csv"
        path.write_text(
            "URL,url_length,ClassLabel\n"
            "https://phish.example/,999,0\n"
            "https://good.example/,999,1\n"
            "https://none.example/,999,\n",
            encoding="utf-8",
        )

        records = list(load_legitphish(path, TIMESTAMP))

        assert [r.label for r in records] == [LABEL_PHISHING, LABEL_LEGITIMATE, None]
        assert records[0].location == "lp.csv:2"

    def test_legitphish_loader_requires_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("link,label\nhttps://a.example/,0\n", encoding="utf-8")

        with pytest.raises(ValueError, match="missing columns"):
            list(load_legitphish(path, TIMESTAMP))


class TestURLCleaning:
    def test_clean_url_unchanged(self) -> None:
        result = clean_url("https://example.com/a?b=1")

        assert result.url == "https://example.com/a?b=1"
        assert result.actions == ()

    def test_trims_whitespace(self) -> None:
        result = clean_url("  https://example.com/ \n")

        assert result.url == "https://example.com/"
        assert result.actions == (ACTION_TRIMMED_WHITESPACE,)

    @pytest.mark.parametrize("quote", ["'", '"'])
    def test_removes_matching_wrapping_quotes(self, quote: str) -> None:
        result = clean_url(f"{quote}example.com/login{quote}")

        assert result.url == "example.com/login"
        assert result.actions == (ACTION_REMOVED_WRAPPING_QUOTES,)

    @pytest.mark.parametrize(
        "value",
        [
            "'example.com/login",  # opening quote only
            "example.com/login'",  # closing quote only
            "'example.com/login\"",  # mismatched pair
            "'example.com/it's'",  # same quote inside: not clearly a wrapper
            "'",  # a lone quote
        ],
    )
    def test_keeps_quotes_that_are_not_a_clear_wrapper(self, value: str) -> None:
        result = clean_url(value)

        assert result.url == value
        assert ACTION_REMOVED_WRAPPING_QUOTES not in result.actions

    def test_unescapes_backslash_before_percent_encoding(self) -> None:
        result = clean_url(r"example.com/paypal\%20us/a\%3Fb")

        assert result.url == "example.com/paypal%20us/a%3Fb"
        assert result.actions == (ACTION_UNESCAPED_PERCENT,)

    @pytest.mark.parametrize(
        "value",
        [
            r"https://example.com/a\b",  # backslash not before %XX
            r"https://example.com/a\%zz",  # not a valid percent-encoding
            r"https://example.com/a\%2",  # truncated percent-encoding
        ],
    )
    def test_leaves_ambiguous_backslashes(self, value: str) -> None:
        result = clean_url(value)

        assert result.url == value
        assert result.actions == ()

    def test_does_not_decode_html_entities(self) -> None:
        url = "https://example.com/?a=1&amp;b=2"
        assert clean_url(url).url == url

    def test_quote_removal_then_unescape_order(self) -> None:
        result = clean_url(r"'example.com/bahi\%20paypal/'")

        assert result.url == "example.com/bahi%20paypal/"
        assert result.actions == (ACTION_REMOVED_WRAPPING_QUOTES, ACTION_UNESCAPED_PERCENT)


class TestURLValidation:
    def test_accepts_domain_and_ip_hosts(self) -> None:
        assert validate_dataset_url("https://example.com/").hostname == "example.com"
        assert validate_dataset_url("http://192.0.2.1/").hostname == "192.0.2.1"

    @pytest.mark.parametrize(
        "value",
        [
            "Offline",  # junk value -> single-label host "offline"
            "http://http://example.com/",  # doubled scheme -> host "http"
            "http://localhost/",
        ],
    )
    def test_rejects_single_label_hosts(self, value: str) -> None:
        with pytest.raises(InvalidURLError, match="not a public domain"):
            validate_dataset_url(value)

    @pytest.mark.parametrize(
        "value", ["ttp://203.0.113.5/x", "example.com/Secure Login.html", "javascript:alert(1)"]
    )
    def test_rejects_urls_the_backend_rejects(self, value: str) -> None:
        with pytest.raises(InvalidURLError):
            validate_dataset_url(value)


class TestPipeline:
    def test_missing_scheme_gets_http_and_is_recorded(self) -> None:
        result = prepare_records([record("example.com/login")])

        row = result.rows[0]
        assert row.url == "http://example.com/login"
        assert row.features["has_https"] == 0
        assert ACTION_ADDED_DEFAULT_SCHEME in row.cleaning_actions
        assert result.cleaning_action_counts[ACTION_ADDED_DEFAULT_SCHEME] == 1

    def test_backend_normalization_is_recorded(self) -> None:
        row = prepare_records([record("HTTPS://Example.COM/a?")]).rows[0]

        assert row.url == "https://example.com/a"
        assert row.cleaning_actions == (ACTION_NORMALIZED_BY_BACKEND,)

    def test_excludes_missing_and_unsupported_labels(self) -> None:
        result = prepare_records(
            [
                record("https://a.example/", label=None, raw_label=""),
                record("https://b.example/", label=None, raw_label="2"),
                record("https://c.example/"),
            ]
        )

        assert [row.domain for row in result.rows] == ["c.example"]
        assert [e.reason for e in result.exclusions] == [
            EXCLUDED_MISSING_LABEL,
            EXCLUDED_UNSUPPORTED_LABEL,
        ]

    def test_excludes_empty_and_invalid_urls(self) -> None:
        result = prepare_records(
            [record(""), record("''"), record("ttp://203.0.113.5/"), record("Offline")]
        )

        assert result.rows == []
        assert [e.reason for e in result.exclusions] == [
            EXCLUDED_EMPTY_URL,
            EXCLUDED_EMPTY_URL,
            EXCLUDED_INVALID_URL,
            EXCLUDED_INVALID_URL,
        ]

    def test_features_are_recomputed_with_backend_extractor(self) -> None:
        row = prepare_records([record("https://login.example.com/verify?a=1")]).rows[0]

        expected = extract_url_features("https://login.example.com/verify?a=1").to_dict()
        assert row.features == expected
        assert tuple(row.features) == FEATURE_NAMES

    def test_counts_inputs_classes_and_sources(self) -> None:
        result = prepare_records(
            [
                record("https://p.example/", LABEL_PHISHING, SOURCE_OPENPHISH),
                record("https://q.example/", LABEL_PHISHING),
                record("https://r.example/", LABEL_LEGITIMATE),
            ]
        )

        assert result.input_counts == {SOURCE_OPENPHISH: 1, SOURCE_LEGITPHISH: 2}
        assert result.class_counts == {LABEL_PHISHING: 2, LABEL_LEGITIMATE: 1}
        assert result.source_counts == {SOURCE_OPENPHISH: 1, SOURCE_LEGITPHISH: 2}


class TestDeduplication:
    def test_exact_duplicates_are_collapsed(self) -> None:
        result = prepare_records([record("https://a.example/x")] * 3)

        assert len(result.rows) == 1
        assert result.exact_duplicates == 2
        assert result.normalized_duplicates == 0

    def test_duplicates_detected_after_normalization(self) -> None:
        result = prepare_records(
            [
                record("https://A.Example/x"),
                record("https://a.example./x"),
                record("'https://a.example/x'"),
                record("HTTPS://a.example/x"),
            ]
        )

        assert [row.url for row in result.rows] == ["https://a.example/x"]
        assert result.normalized_duplicates == 3

    def test_scheme_less_and_http_forms_are_duplicates(self) -> None:
        result = prepare_records([record("a.example/x"), record("http://a.example/x")])

        assert len(result.rows) == 1

    def test_http_and_https_are_different_urls(self) -> None:
        result = prepare_records([record("http://a.example/"), record("https://a.example/")])

        assert len(result.rows) == 2

    def test_first_occurrence_kept_and_all_sources_recorded(self) -> None:
        result = prepare_records(
            [
                record("https://a.example/", source=SOURCE_OPENPHISH, collected_at="2026-02-01T00:00:00Z"),
                record("https://a.example/", source=SOURCE_LEGITPHISH, collected_at="2026-01-01T00:00:00Z"),
            ]
        )

        row = result.rows[0]
        assert row.source == "legitphish;openphish"
        assert row.collected_at == "2026-01-01T00:00:00Z"
        assert result.cross_source_duplicates == 1

    def test_output_order_follows_first_occurrence(self) -> None:
        result = prepare_records(
            [record("https://b.example/"), record("https://a.example/"), record("https://b.example/")]
        )

        assert [row.domain for row in result.rows] == ["b.example", "a.example"]


class TestConflictingLabels:
    def test_conflicting_url_is_excluded_and_recorded(self) -> None:
        result = prepare_records(
            [
                record("https://a.example/", LABEL_PHISHING, SOURCE_OPENPHISH),
                record("https://a.example/", LABEL_LEGITIMATE, SOURCE_LEGITPHISH),
                record("https://b.example/", LABEL_LEGITIMATE),
            ]
        )

        assert [row.domain for row in result.rows] == ["b.example"]
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.url == "https://a.example/"
        assert conflict.labels == (LABEL_LEGITIMATE, LABEL_PHISHING)
        assert len(conflict.records) == 2
        reasons = [e.reason for e in result.exclusions]
        assert reasons == [EXCLUDED_CONFLICTING_LABELS, EXCLUDED_CONFLICTING_LABELS]

    def test_conflict_detected_after_normalization(self) -> None:
        result = prepare_records(
            [
                record("'example.com/login'", LABEL_PHISHING),
                record("HTTP://EXAMPLE.COM/login", LABEL_LEGITIMATE),
            ]
        )

        assert result.rows == []
        assert result.conflicts[0].url == "http://example.com/login"

    def test_conflict_excludes_every_copy_including_agreeing_duplicates(self) -> None:
        result = prepare_records(
            [
                record("https://a.example/", LABEL_PHISHING),
                record("https://a.example/", LABEL_PHISHING),
                record("https://a.example/", LABEL_LEGITIMATE),
            ]
        )

        assert result.rows == []
        assert len(result.conflicts[0].records) == 3
        assert result.exact_duplicates == 0


class TestOutput:
    def test_write_dataset_columns_and_values(self, tmp_path: Path) -> None:
        result = prepare_records(
            [record("example.com/login", LABEL_PHISHING, SOURCE_OPENPHISH)]
        )
        path = tmp_path / "out" / "dataset.csv"

        assert write_dataset(result.rows, path) == 1

        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS
        assert rows[0]["url"] == "http://example.com/login"
        assert rows[0]["domain"] == "example.com"
        assert rows[0]["label"] == "1"
        assert rows[0]["source"] == SOURCE_OPENPHISH
        assert rows[0]["collected_at"] == TIMESTAMP
        assert rows[0]["has_suspicious_keywords"] == "1"
        assert rows[0]["cleaning_actions"] == ACTION_ADDED_DEFAULT_SCHEME
