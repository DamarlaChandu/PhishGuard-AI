"""Inspect the raw PhishGuard AI datasets and write a plain-text report.

What it does:
  * Verifies and safely extracts the LegitPhish and Tranco ZIP archives into
    data/interim/ (originals in data/raw/ are only read, never changed).
  * Reports structure and data-quality findings for each CSV.
  * Checks the LegitPhish label direction against independent evidence
    (OpenPhish feed overlap and top-ranked Tranco domains).
  * Writes data/processed/dataset_inspection_report.txt.

It does not merge datasets, train models, or make any network requests.
Tranco domains are reported only as *candidate* legitimate domains.

Usage (from the project root):
    python scripts/inspect_datasets.py
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.features import InvalidURLError, extract_url_features, normalize_url  # noqa: E402
from app.ml.features.extractor import is_shortener_host  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_inspection_report.txt"

LEGITPHISH_ZIP = RAW_DIR / "LegitPhish Dataset.zip"
TRANCO_ZIP = RAW_DIR / "tranco_GQ9JK-1m.csv.zip"
OPENPHISH_FEED = RAW_DIR / "openphish_feed.txt"

LEGITPHISH_URL_COLUMN = "URL"
LEGITPHISH_LABEL_COLUMN = "ClassLabel"

# Refuse archives that would expand beyond this (zip-bomb guard).
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024**3
# Tranco ranks considered "highly popular" when checking label direction.
TOP_TRANCO_RANK = 10_000
SAMPLE_SIZE = 5


# --------------------------------------------------------------------------- #
# Archive handling
# --------------------------------------------------------------------------- #
@dataclass
class ArchiveInfo:
    path: Path
    members: list[tuple[str, int, int]]  # (name, uncompressed, compressed)
    extracted_to: Path
    extracted_files: list[Path]


def safe_extract(zip_path: Path, target_dir: Path) -> ArchiveInfo:
    """Integrity-check ``zip_path`` and extract it into ``target_dir``.

    Every member path must resolve inside ``target_dir``; absolute paths,
    ``..`` traversal and oversized archives are rejected before writing.
    """
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"{zip_path.name}: CRC check failed for {corrupt!r}")

        infos = archive.infolist()
        total = sum(info.file_size for info in infos)
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise RuntimeError(f"{zip_path.name}: uncompressed size {total} exceeds limit")

        for info in infos:
            destination = (target_root / info.filename).resolve()
            if not destination.is_relative_to(target_root):
                raise RuntimeError(f"{zip_path.name}: unsafe member path {info.filename!r}")

        target_root.mkdir(parents=True, exist_ok=True)
        archive.extractall(target_root)

        members = [(i.filename, i.file_size, i.compress_size) for i in infos]
        extracted = [target_root / i.filename for i in infos if not i.is_dir()]
    return ArchiveInfo(zip_path, members, target_root, extracted)


def find_single_csv(info: ArchiveInfo) -> Path:
    csv_files = [p for p in info.extracted_files if p.suffix.lower() == ".csv"]
    if len(csv_files) != 1:
        raise RuntimeError(f"{info.path.name}: expected one CSV, found {csv_files}")
    return csv_files[0]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def url_key(url: str) -> str | None:
    """Normalized URL for duplicate/overlap matching, or None if invalid."""
    try:
        return normalize_url(url).url
    except InvalidURLError:
        return None


def hostname_of(url: str) -> str | None:
    try:
        return normalize_url(url).hostname
    except InvalidURLError:
        return None


def registrable_host(hostname: str) -> str:
    return hostname.removeprefix("www.")


def read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def duplicate_stats(values: list[str]) -> tuple[int, int]:
    """Return (extra duplicate rows, distinct values that occur more than once)."""
    counts = Counter(values)
    repeated = [c for c in counts.values() if c > 1]
    return sum(c - 1 for c in repeated), len(repeated)


# --------------------------------------------------------------------------- #
# LegitPhish
# --------------------------------------------------------------------------- #
@dataclass
class LegitPhishFindings:
    csv_path: Path
    columns: list[str]
    total_rows: int = 0
    malformed_rows: int = 0
    label_counts: Counter = field(default_factory=Counter)
    missing_labels: int = 0
    missing_label_examples: list[str] = field(default_factory=list)
    unexpected_labels: Counter = field(default_factory=Counter)
    empty_urls: int = 0
    quoted_urls: Counter = field(default_factory=Counter)  # by label
    backslash_urls: Counter = field(default_factory=Counter)  # by label
    schemeless_urls: Counter = field(default_factory=Counter)  # by label
    top_hosts: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    invalid_urls: int = 0
    invalid_url_examples: list[str] = field(default_factory=list)
    exact_duplicate_rows: int = 0
    exact_duplicate_urls: int = 0
    normalized_duplicate_rows: int = 0
    normalized_duplicate_urls: int = 0
    conflicting_label_urls: int = 0
    conflicting_examples: list[str] = field(default_factory=list)
    samples_by_label: dict[str, list[str]] = field(default_factory=dict)
    rows: list[tuple[str, str]] = field(default_factory=list)  # (url, label)
    line_numbers: list[int] = field(default_factory=list)  # CSV line of each row


def inspect_legitphish(csv_path: Path) -> LegitPhishFindings:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        findings = LegitPhishFindings(csv_path, columns)
        url_idx = columns.index(LEGITPHISH_URL_COLUMN)
        label_idx = columns.index(LEGITPHISH_LABEL_COLUMN)

        for row in reader:
            if not row:
                continue
            findings.total_rows += 1
            if len(row) != len(columns):
                findings.malformed_rows += 1
            url = row[url_idx].strip() if len(row) > url_idx else ""
            label = row[label_idx].strip() if len(row) > label_idx else ""
            findings.rows.append((url, label))
            findings.line_numbers.append(reader.line_num)

    urls_exact: list[str] = []
    urls_normalized: list[str] = []
    labels_by_key: dict[str, set[str]] = {}
    hosts_by_label: dict[str, Counter] = {"0": Counter(), "1": Counter()}

    for (url, label), line_no in zip(findings.rows, findings.line_numbers):
        if url[:1] in "'\"" or url[-1:] in "'\"":
            findings.quoted_urls[label] += 1
        if "\\" in url:
            findings.backslash_urls[label] += 1
        if url and not url.lower().startswith(("http://", "https://")):
            findings.schemeless_urls[label] += 1

        if label == "":
            findings.missing_labels += 1
            if len(findings.missing_label_examples) < SAMPLE_SIZE:
                findings.missing_label_examples.append(f"CSV line {line_no}: {url}")
        elif label in {"0", "1"}:
            findings.label_counts[label] += 1
            samples = findings.samples_by_label.setdefault(label, [])
            if len(samples) < SAMPLE_SIZE:
                samples.append(url)
        else:
            findings.unexpected_labels[label] += 1

        if not url:
            findings.empty_urls += 1
            continue
        urls_exact.append(url)
        key = url_key(url)
        if key is None:
            findings.invalid_urls += 1
            if len(findings.invalid_url_examples) < 10:
                findings.invalid_url_examples.append(url)
            continue
        urls_normalized.append(key)
        if label:
            labels_by_key.setdefault(key, set()).add(label)
        if label in hosts_by_label:
            host = hostname_of(url)
            if host is not None:
                hosts_by_label[label][registrable_host(host)] += 1

    findings.top_hosts = {
        label: counter.most_common(SAMPLE_SIZE) for label, counter in hosts_by_label.items()
    }

    findings.exact_duplicate_rows, findings.exact_duplicate_urls = duplicate_stats(urls_exact)
    (
        findings.normalized_duplicate_rows,
        findings.normalized_duplicate_urls,
    ) = duplicate_stats(urls_normalized)

    conflicting = [k for k, labels in labels_by_key.items() if len(labels) > 1]
    findings.conflicting_label_urls = len(conflicting)
    findings.conflicting_examples = conflicting[:SAMPLE_SIZE]
    return findings


# --------------------------------------------------------------------------- #
# Tranco
# --------------------------------------------------------------------------- #
@dataclass
class TrancoFindings:
    csv_path: Path
    has_header: bool
    column_layout: str
    total_rows: int = 0
    malformed_rows: int = 0
    empty_domains: int = 0
    invalid_domains: int = 0
    invalid_examples: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    duplicate_domains: int = 0
    rank_issues: int = 0
    first_rank: int | None = None
    last_rank: int | None = None
    samples_head: list[tuple[str, str]] = field(default_factory=list)
    samples_tail: list[tuple[str, str]] = field(default_factory=list)
    rank_by_domain: dict[str, int] = field(default_factory=dict)


def is_valid_domain(domain: str) -> bool:
    """A bare domain must parse as a URL host and round-trip unchanged."""
    try:
        parsed = normalize_url(f"http://{domain}/")
    except InvalidURLError:
        return False
    return parsed.hostname == domain.lower() and parsed.port is None


def inspect_tranco(csv_path: Path) -> TrancoFindings:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    first = rows[0] if rows else []
    has_header = bool(first) and not first[0].strip().isdigit()
    data_rows = rows[1:] if has_header else rows
    findings = TrancoFindings(
        csv_path,
        has_header,
        "header row present" if has_header else "no header; columns are rank,domain",
    )

    domains: list[str] = []
    expected_rank = 1
    for row in data_rows:
        if not row:
            continue
        findings.total_rows += 1
        if len(row) != 2:
            findings.malformed_rows += 1
            continue
        rank_text, domain = row[0].strip(), row[1].strip()

        if rank_text.isdigit():
            rank = int(rank_text)
            findings.first_rank = rank if findings.first_rank is None else findings.first_rank
            findings.last_rank = rank
            if rank != expected_rank:
                findings.rank_issues += 1
            expected_rank = rank + 1
        else:
            findings.rank_issues += 1
            rank = -1

        if not domain:
            findings.empty_domains += 1
            continue
        if not is_valid_domain(domain):
            findings.invalid_domains += 1
            if len(findings.invalid_examples) < 10:
                findings.invalid_examples.append(domain)
        key = domain.lower()
        domains.append(key)
        findings.rank_by_domain.setdefault(key, rank)

    findings.duplicate_rows, findings.duplicate_domains = duplicate_stats(domains)
    findings.samples_head = [(r[0], r[1]) for r in data_rows[:SAMPLE_SIZE]]
    findings.samples_tail = [(r[0], r[1]) for r in data_rows[-SAMPLE_SIZE:]]
    return findings


# --------------------------------------------------------------------------- #
# OpenPhish (read-only, used as label-direction evidence)
# --------------------------------------------------------------------------- #
@dataclass
class OpenPhishFindings:
    total_lines: int
    non_empty: int
    valid: int
    duplicate_rows: int
    keys: set[str]


def inspect_openphish(path: Path) -> OpenPhishFindings:
    lines = read_text_lines(path)
    entries = [line.strip() for line in lines if line.strip()]
    keys = [k for k in (url_key(e) for e in entries) if k is not None]
    dup_rows, _ = duplicate_stats(entries)
    return OpenPhishFindings(len(lines), len(entries), len(keys), dup_rows, set(keys))


# --------------------------------------------------------------------------- #
# Label-direction evidence
# --------------------------------------------------------------------------- #
@dataclass
class LabelStats:
    rows: int = 0
    keyword_rows: int = 0
    distinct_hosts: int = 0  # excluding URL shorteners
    hosts_in_tranco_1m: int = 0
    hosts_in_tranco_top: int = 0
    openphish_matches: int = 0


@dataclass
class LabelEvidence:
    stats: dict[str, LabelStats]
    conclusion: str


# A label "looks legitimate" when its distinct hosts appear in Tranco at least
# this many times more often than the other label's hosts.
POPULARITY_RATIO_THRESHOLD = 5.0


def tranco_rate(stats: LabelStats) -> float:
    return stats.hosts_in_tranco_1m / stats.distinct_hosts if stats.distinct_hosts else 0.0


def assess_label_direction(
    legit: LegitPhishFindings, tranco: TrancoFindings, openphish: OpenPhishFindings
) -> LabelEvidence:
    """Compare each label against independent popularity and phishing signals.

    Distinct hosts are compared (not rows), because a few hosts contribute
    hundreds of rows each. URL shorteners are excluded, since popular
    shortener domains carry phishing links too.
    """
    stats = {"0": LabelStats(), "1": LabelStats()}
    hosts: dict[str, set[str]] = {"0": set(), "1": set()}

    for url, label in legit.rows:
        if label not in stats:
            continue
        try:
            parsed = normalize_url(url)
        except InvalidURLError:
            continue
        label_stats = stats[label]
        label_stats.rows += 1
        if parsed.url in openphish.keys:
            label_stats.openphish_matches += 1
        if extract_url_features(url).has_suspicious_keywords:
            label_stats.keyword_rows += 1
        host = registrable_host(parsed.hostname)
        if not is_shortener_host(host):
            hosts[label].add(host)

    for label, label_hosts in hosts.items():
        label_stats = stats[label]
        label_stats.distinct_hosts = len(label_hosts)
        for host in label_hosts:
            rank = tranco.rank_by_domain.get(host)
            if rank is not None:
                label_stats.hosts_in_tranco_1m += 1
                if 0 < rank <= TOP_TRANCO_RANK:
                    label_stats.hosts_in_tranco_top += 1

    rate0, rate1 = tranco_rate(stats["0"]), tranco_rate(stats["1"])
    openphish_against = stats["1"].openphish_matches > stats["0"].openphish_matches
    if rate1 >= POPULARITY_RATIO_THRESHOLD * rate0 and rate1 > 0 and not openphish_against:
        conclusion = (
            "SUPPORTED: 0 = Phishing and 1 = Legitimate. Label-1 hosts are found in "
            f"Tranco {rate1 / rate0 if rate0 else float('inf'):.0f}x more often than "
            "label-0 hosts, and label 0 shows more lure keywords. This is strong "
            "statistical evidence, not a per-row guarantee; cite the dataset's "
            "original documentation as the authoritative source."
        )
    elif rate0 >= POPULARITY_RATIO_THRESHOLD * rate1 and rate0 > 0:
        conclusion = "CONTRADICTED: the evidence points to 1 = Phishing and 0 = Legitimate."
    else:
        conclusion = (
            "INCONCLUSIVE: popularity rates do not separate the labels clearly. "
            "Confirm the mapping against the dataset's original documentation."
        )
    return LabelEvidence(stats, conclusion)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def pct(part: int, whole: int) -> str:
    return f"{(100 * part / whole):.2f}%" if whole else "n/a"


def format_report(
    archives: list[ArchiveInfo],
    legit: LegitPhishFindings,
    tranco: TrancoFindings,
    openphish: OpenPhishFindings,
    evidence: LabelEvidence,
) -> str:
    out: list[str] = []
    add = out.append
    rule = "=" * 78

    add(rule)
    add("PhishGuard AI - Dataset Inspection Report")
    add(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    add("Generator: scripts/inspect_datasets.py")
    add("Scope:     inspection only. No merging, labelling changes or training.")
    add(rule)

    add("\n1. ARCHIVES")
    for info in archives:
        add(f"  {rel(info.path)}  (CRC check: OK)")
        for name, size, compressed in info.members:
            add(f"    member: {name!r}  {size:,} bytes ({compressed:,} compressed)")
        add(f"    extracted to: {rel(info.extracted_to)}/")
    add("  Original files in data/raw/ were opened read-only and not modified.")

    total = legit.total_rows
    labelled = sum(legit.label_counts.values())
    add("\n2. LEGITPHISH")
    add(f"  CSV file:        {rel(legit.csv_path)}")
    add(f"  Columns ({len(legit.columns)}):")
    for col in legit.columns:
        add(f"    - {col}")
    add(f"  URL column:      {LEGITPHISH_URL_COLUMN}")
    add(f"  Label column:    {LEGITPHISH_LABEL_COLUMN}")
    add(f"  Total rows:      {total:,}")
    add(f"  Malformed rows (wrong column count): {legit.malformed_rows:,}")
    add("  Label counts:")
    add(f"    0 (Phishing, per stated mapping):   {legit.label_counts['0']:,} ({pct(legit.label_counts['0'], total)})")
    add(f"    1 (Legitimate, per stated mapping): {legit.label_counts['1']:,} ({pct(legit.label_counts['1'], total)})")
    add(f"  Missing labels:  {legit.missing_labels:,}")
    for example in legit.missing_label_examples:
        add(f"    {example}")
    unexpected = dict(legit.unexpected_labels) or "none"
    add(f"  Labels other than 0/1: {unexpected}")
    add(f"  Empty URLs:      {legit.empty_urls:,}")
    add(f"  Invalid URLs (rejected by backend normalize_url): {legit.invalid_urls:,}")
    for example in legit.invalid_url_examples:
        add(f"    e.g. {example}")
    add("  URL formatting artifacts (label 0 / label 1):")
    add(f"    wrapped in quote characters: {legit.quoted_urls['0']:,} / {legit.quoted_urls['1']:,}")
    add(f"    contain a backslash:         {legit.backslash_urls['0']:,} / {legit.backslash_urls['1']:,}")
    add(f"    no http(s):// scheme:        {legit.schemeless_urls['0']:,} / {legit.schemeless_urls['1']:,}")
    add("    A label-specific artifact can let a model learn formatting instead")
    add("    of phishing signals; clean or account for it before training.")
    add(f"  Duplicate URLs (exact string match): {legit.exact_duplicate_rows:,} extra rows "
        f"across {legit.exact_duplicate_urls:,} distinct URLs")
    add(f"  Duplicate URLs (after normalization): {legit.normalized_duplicate_rows:,} extra rows "
        f"across {legit.normalized_duplicate_urls:,} distinct URLs")
    add(f"  URLs with conflicting labels (same normalized URL labelled 0 and 1): "
        f"{legit.conflicting_label_urls:,}")
    for example in legit.conflicting_examples:
        add(f"    e.g. {example}")
    for label in ("0", "1"):
        add(f"  Sample URLs with label {label}:")
        for url in legit.samples_by_label.get(label, []):
            add(f"    {url}")
    for label in ("0", "1"):
        add(f"  Most frequent hosts with label {label} (host: rows):")
        for host, count in legit.top_hosts.get(label, []):
            add(f"    {host}: {count:,}")
    add("    Rows sharing a host must stay in the same train/test split to")
    add("    avoid leakage.")
    add(f"  Rows with a usable 0/1 label: {labelled:,}")

    s0, s1 = evidence.stats["0"], evidence.stats["1"]
    add("\n3. LEGITPHISH LABEL DIRECTION (0 = Phishing, 1 = Legitimate?)")
    add("  The CSV has no label legend, so the mapping was checked against")
    add("  independent sources (valid URLs only; hosts exclude 'www.' and URL")
    add("  shorteners):")
    add(f"  {'':42}{'label 0':>14}{'label 1':>14}")
    add(f"  {'Valid labelled URLs':42}{s0.rows:>14,}{s1.rows:>14,}")
    add(f"  {'Distinct hosts':42}{s0.distinct_hosts:>14,}{s1.distinct_hosts:>14,}")
    add(f"  {'Hosts in Tranco top 1M':42}{s0.hosts_in_tranco_1m:>14,}{s1.hosts_in_tranco_1m:>14,}")
    add(f"  {'  as % of distinct hosts':42}{pct(s0.hosts_in_tranco_1m, s0.distinct_hosts):>14}"
        f"{pct(s1.hosts_in_tranco_1m, s1.distinct_hosts):>14}")
    add(f"  {f'Hosts in Tranco top {TOP_TRANCO_RANK:,}':42}{s0.hosts_in_tranco_top:>14,}{s1.hosts_in_tranco_top:>14,}")
    add(f"  {'URLs with a suspicious keyword':42}{pct(s0.keyword_rows, s0.rows):>14}"
        f"{pct(s1.keyword_rows, s1.rows):>14}")
    add(f"  {'Exact matches in OpenPhish feed':42}{s0.openphish_matches:>14,}{s1.openphish_matches:>14,}")
    if s0.openphish_matches == s1.openphish_matches == 0:
        add("  (No LegitPhish URL appears in the OpenPhish feed snapshot, so that")
        add("  check provides no evidence either way.)")
    add(f"  Conclusion: {evidence.conclusion}")

    add("\n4. TRANCO")
    add(f"  CSV file:        {rel(tranco.csv_path)}")
    add(f"  Layout:          {tranco.column_layout}")
    add(f"  Total domains:   {tranco.total_rows:,}")
    add(f"  Rank range:      {tranco.first_rank} .. {tranco.last_rank}")
    add(f"  Rank sequence breaks (non-consecutive or non-numeric): {tranco.rank_issues:,}")
    add(f"  Malformed rows (not exactly 2 columns): {tranco.malformed_rows:,}")
    add(f"  Empty domains:   {tranco.empty_domains:,}")
    add(f"  Invalid domains: {tranco.invalid_domains:,}")
    for example in tranco.invalid_examples:
        add(f"    e.g. {example!r}")
    add(f"  Duplicate domains (case-insensitive): {tranco.duplicate_rows:,} extra rows "
        f"across {tranco.duplicate_domains:,} distinct domains")
    add("  Sample records (first):")
    for rank, domain in tranco.samples_head:
        add(f"    {rank:>7}  {domain}")
    add("  Sample records (last):")
    for rank, domain in tranco.samples_tail:
        add(f"    {rank:>7}  {domain}")
    add("  NOTE: Tranco ranks domains by popularity. Popularity is not proof of")
    add("  safety (popular hosting platforms can serve phishing pages), so these")
    add("  are treated only as CANDIDATE legitimate domains and are NOT labelled.")

    add("\n5. OPENPHISH FEED (read-only, used for evidence only)")
    add(f"  File:            {rel(OPENPHISH_FEED)}")
    add(f"  Lines:           {openphish.total_lines:,}")
    add(f"  Non-empty:       {openphish.non_empty:,}")
    add(f"  Valid URLs:      {openphish.valid:,}")
    add(f"  Duplicate lines: {openphish.duplicate_rows:,}")

    add("\n" + rule)
    return "\n".join(out) + "\n"


def main() -> None:
    legit_archive = safe_extract(LEGITPHISH_ZIP, INTERIM_DIR / "legitphish")
    tranco_archive = safe_extract(TRANCO_ZIP, INTERIM_DIR / "tranco")

    legit = inspect_legitphish(find_single_csv(legit_archive))
    tranco = inspect_tranco(find_single_csv(tranco_archive))
    openphish = inspect_openphish(OPENPHISH_FEED)
    evidence = assess_label_direction(legit, tranco, openphish)

    report = format_report([legit_archive, tranco_archive], legit, tranco, openphish, evidence)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    stdout.write(report)
    stdout.write(f"Report written to {rel(REPORT_PATH)}\n")
    stdout.flush()


if __name__ == "__main__":
    main()
