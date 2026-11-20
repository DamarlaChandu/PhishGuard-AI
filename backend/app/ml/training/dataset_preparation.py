"""Build a labelled URL dataset from raw phishing and legitimate sources.

Project label convention: 1 = phishing, 0 = legitimate.

Pipeline for every raw record:
  1. Convert the source label to the project convention (or exclude it).
  2. Clean the URL conservatively (see :func:`clean_url`).
  3. Validate it with the backend's :func:`normalize_url`, plus a dataset rule
     that the host must be an IP address or contain a dot.
  4. Group by normalized URL: identical labels are deduplicated, conflicting
     labels are excluded and recorded.
  5. Recompute the 15 URL features with the backend extractor.

Nothing here fetches URLs, resolves hosts or executes content.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.ml.features import (
    FEATURE_NAMES,
    InvalidURLError,
    NormalizedURL,
    extract_features_from_normalized,
    normalize_url,
)
from app.ml.features.ip_address import parse_ip_hostname

LABEL_PHISHING = 1
LABEL_LEGITIMATE = 0
LABEL_NAMES = {LABEL_PHISHING: "phishing", LABEL_LEGITIMATE: "legitimate"}

SOURCE_OPENPHISH = "openphish"
SOURCE_LEGITPHISH = "legitphish"

# LegitPhish uses 0 = Phishing, 1 = Legitimate; the project uses the reverse.
LEGITPHISH_LABEL_MAP: dict[str, int] = {"0": LABEL_PHISHING, "1": LABEL_LEGITIMATE}
LEGITPHISH_URL_COLUMN = "URL"
LEGITPHISH_LABEL_COLUMN = "ClassLabel"

OUTPUT_COLUMNS: tuple[str, ...] = (
    "url",
    "domain",
    "label",
    "source",
    "collected_at",
    *FEATURE_NAMES,
    "cleaning_actions",
)

# Cleaning action names, recorded per row and counted in the report.
ACTION_TRIMMED_WHITESPACE = "trimmed_whitespace"
ACTION_REMOVED_WRAPPING_QUOTES = "removed_wrapping_quotes"
ACTION_UNESCAPED_PERCENT = "unescaped_backslash_percent"
ACTION_ADDED_DEFAULT_SCHEME = "added_default_scheme"
# Backend normalization changed the string (lowercased scheme/host, dropped a
# trailing host dot, an empty port or an empty "?" query).
ACTION_NORMALIZED_BY_BACKEND = "normalized_by_backend"

# Exclusion reasons.
EXCLUDED_MISSING_LABEL = "missing_label"
EXCLUDED_UNSUPPORTED_LABEL = "unsupported_label"
EXCLUDED_EMPTY_URL = "empty_url"
EXCLUDED_INVALID_URL = "invalid_url"
EXCLUDED_CONFLICTING_LABELS = "conflicting_labels"

_QUOTE_CHARS = ("'", '"')
# A backslash immediately before a percent-encoded byte, e.g. "\%20".
_ESCAPED_PERCENT_RE = re.compile(r"\\(%[0-9A-Fa-f]{2})")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RawRecord:
    """One input row before cleaning.

    ``label`` is already in the project convention, or None if the source
    label was missing or unsupported (``raw_label`` keeps the original text).
    """

    source: str
    raw_url: str
    raw_label: str
    label: int | None
    collected_at: str
    location: str  # e.g. "openphish_feed.txt:12" for traceability


@dataclass(frozen=True, slots=True)
class CleanedURL:
    url: str
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetRow:
    url: str
    domain: str
    label: int
    source: str
    collected_at: str
    features: dict[str, int]
    cleaning_actions: tuple[str, ...]

    def to_csv_row(self) -> dict[str, str | int]:
        return {
            "url": self.url,
            "domain": self.domain,
            "label": self.label,
            "source": self.source,
            "collected_at": self.collected_at,
            **self.features,
            "cleaning_actions": ";".join(self.cleaning_actions),
        }


@dataclass(frozen=True, slots=True)
class Exclusion:
    record: RawRecord
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Conflict:
    url: str
    records: tuple[RawRecord, ...]

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(sorted({r.label for r in self.records if r.label is not None}))


@dataclass
class PreparationResult:
    rows: list[DatasetRow] = field(default_factory=list)
    input_counts: Counter = field(default_factory=Counter)  # by source
    exclusions: list[Exclusion] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    cleaning_action_counts: Counter = field(default_factory=Counter)
    cleaning_examples: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    exact_duplicates: int = 0  # same cleaned string as an earlier record
    normalized_duplicates: int = 0  # different string, same normalized URL
    cross_source_duplicates: int = 0  # duplicate groups spanning sources

    @property
    def class_counts(self) -> Counter:
        return Counter(row.label for row in self.rows)

    @property
    def source_counts(self) -> Counter:
        return Counter(row.source for row in self.rows)

    def exclusion_counts(self) -> Counter:
        return Counter((e.record.source, e.reason) for e in self.exclusions)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def convert_legitphish_label(raw_label: str | None) -> int | None:
    """Map a LegitPhish ClassLabel to the project label, or None if unusable.

    "0" (Phishing) -> 1, "1" (Legitimate) -> 0. Anything else, including
    empty values and "0.0"-style variants, is unsupported.
    """
    if raw_label is None:
        return None
    return LEGITPHISH_LABEL_MAP.get(raw_label.strip())


# --------------------------------------------------------------------------- #
# Cleaning and validation
# --------------------------------------------------------------------------- #
def clean_url(raw_url: str) -> CleanedURL:
    """Apply only unambiguous, content-preserving fixes to a raw URL string.

    * Trim surrounding whitespace.
    * Remove one pair of matching quote characters wrapping the whole value,
      only when that quote character does not also appear inside the URL
      (a CSV/SQL export artifact).
    * Turn ``\\%XX`` into ``%XX``: a backslash-escaped percent-encoded byte.
      Other backslashes are left untouched.

    Nothing else is rewritten: HTML entities such as ``&amp;``, inner
    whitespace and truncated schemes are left for validation to judge.
    """
    actions: list[str] = []
    text = raw_url

    stripped = text.strip()
    if stripped != text:
        actions.append(ACTION_TRIMMED_WHITESPACE)
        text = stripped

    if (
        len(text) >= 2
        and text[0] in _QUOTE_CHARS
        and text[-1] == text[0]
        and text[0] not in text[1:-1]
    ):
        actions.append(ACTION_REMOVED_WRAPPING_QUOTES)
        text = text[1:-1].strip()

    unescaped = _ESCAPED_PERCENT_RE.sub(r"\1", text)
    if unescaped != text:
        actions.append(ACTION_UNESCAPED_PERCENT)
        text = unescaped

    return CleanedURL(text, tuple(actions))


def validate_dataset_url(url: str) -> NormalizedURL:
    """Validate with the backend normalizer plus dataset-specific rules.

    Raises:
        InvalidURLError: if the backend rejects the URL, or the host is a
            single label (e.g. "offline", "localhost") rather than an IP or a
            dotted domain name. Such hosts are not public web addresses.
    """
    parsed = normalize_url(url)
    if "." not in parsed.hostname and parse_ip_hostname(parsed.hostname) is None:
        raise InvalidURLError(f"Hostname is not a public domain or IP: {parsed.hostname!r}")
    return parsed


def _normalization_actions(cleaned: str, parsed: NormalizedURL) -> list[str]:
    actions = []
    if not parsed.scheme_was_explicit:
        actions.append(ACTION_ADDED_DEFAULT_SCHEME)
    elif parsed.url != cleaned:
        actions.append(ACTION_NORMALIZED_BY_BACKEND)
    return actions


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class _Candidate:
    record: RawRecord
    label: int
    cleaned: str
    parsed: NormalizedURL
    actions: tuple[str, ...]


_MAX_EXAMPLES = 5


def prepare_records(records: Iterable[RawRecord]) -> PreparationResult:
    """Clean, validate, deduplicate and featurize records, in input order."""
    result = PreparationResult()
    groups: dict[str, list[_Candidate]] = {}

    for record in records:
        result.input_counts[record.source] += 1
        candidate = _process_record(record, result)
        if candidate is not None:
            groups.setdefault(candidate.parsed.url, []).append(candidate)

    for url, candidates in groups.items():
        labels = {c.label for c in candidates}
        if len(labels) > 1:
            result.conflicts.append(Conflict(url, tuple(c.record for c in candidates)))
            for candidate in candidates:
                result.exclusions.append(
                    Exclusion(candidate.record, EXCLUDED_CONFLICTING_LABELS, url)
                )
            continue
        _count_duplicates(candidates, result)
        result.rows.append(_build_row(candidates))

    return result


def _process_record(record: RawRecord, result: PreparationResult) -> _Candidate | None:
    if record.label is None:
        reason = (
            EXCLUDED_MISSING_LABEL if not record.raw_label.strip() else EXCLUDED_UNSUPPORTED_LABEL
        )
        result.exclusions.append(Exclusion(record, reason, record.raw_label))
        return None

    cleaned = clean_url(record.raw_url)
    if not cleaned.url:
        result.exclusions.append(Exclusion(record, EXCLUDED_EMPTY_URL))
        return None

    try:
        parsed = validate_dataset_url(cleaned.url)
    except InvalidURLError as exc:
        result.exclusions.append(Exclusion(record, EXCLUDED_INVALID_URL, str(exc)))
        return None

    actions = (*cleaned.actions, *_normalization_actions(cleaned.url, parsed))
    for action in actions:
        result.cleaning_action_counts[action] += 1
        examples = result.cleaning_examples.setdefault(action, [])
        if len(examples) < _MAX_EXAMPLES:
            examples.append((record.raw_url, parsed.url))
    return _Candidate(record, record.label, cleaned.url, parsed, actions)


def _count_duplicates(candidates: list[_Candidate], result: PreparationResult) -> None:
    seen_strings: set[str] = set()
    for index, candidate in enumerate(candidates):
        if index == 0:
            seen_strings.add(candidate.cleaned)
            continue
        if candidate.cleaned in seen_strings:
            result.exact_duplicates += 1
        else:
            result.normalized_duplicates += 1
            seen_strings.add(candidate.cleaned)
    if len({c.record.source for c in candidates}) > 1:
        result.cross_source_duplicates += 1


def _build_row(candidates: list[_Candidate]) -> DatasetRow:
    """Keep the first occurrence, recording every source that supplied the URL."""
    first = candidates[0]
    sources = sorted({c.record.source for c in candidates})
    features = extract_features_from_normalized(first.parsed).to_dict()
    return DatasetRow(
        url=first.parsed.url,
        domain=first.parsed.hostname,
        label=first.label,
        source=";".join(sources),
        collected_at=min(c.record.collected_at for c in candidates),
        features=features,
        cleaning_actions=first.actions,
    )


# --------------------------------------------------------------------------- #
# Loaders and writer
# --------------------------------------------------------------------------- #
def file_timestamp(path: Path) -> str:
    """UTC ISO-8601 modification time of ``path`` (when the file was obtained)."""
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return modified.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_openphish(path: Path, collected_at: str) -> Iterator[RawRecord]:
    """Yield one phishing record per non-blank line of an OpenPhish feed."""
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield RawRecord(
                source=SOURCE_OPENPHISH,
                raw_url=line.rstrip("\r\n"),
                raw_label="phishing (feed)",
                label=LABEL_PHISHING,
                collected_at=collected_at,
                location=f"{path.name}:{line_no}",
            )


def load_legitphish(path: Path, collected_at: str) -> Iterator[RawRecord]:
    """Yield records from the LegitPhish CSV, reading only the URL and label.

    The 16 precomputed feature columns are deliberately ignored.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {LEGITPHISH_URL_COLUMN, LEGITPHISH_LABEL_COLUMN} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row in reader:
            raw_label = row.get(LEGITPHISH_LABEL_COLUMN) or ""
            yield RawRecord(
                source=SOURCE_LEGITPHISH,
                raw_url=row.get(LEGITPHISH_URL_COLUMN) or "",
                raw_label=raw_label,
                label=convert_legitphish_label(raw_label),
                collected_at=collected_at,
                location=f"{path.name}:{reader.line_num}",
            )


def write_dataset(rows: Iterable[DatasetRow], path: Path) -> int:
    """Write rows to CSV with :data:`OUTPUT_COLUMNS`; return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
            count += 1
    return count
