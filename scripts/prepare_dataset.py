"""Prepare the PhishGuard AI training dataset from raw sources.

Inputs (read-only):
  * data/raw/openphish_feed.txt                       -> phishing (label 1)
  * data/interim/legitphish/**/<csv>                  -> LegitPhish labels, converted
    (extract it first with scripts/inspect_datasets.py)

Outputs:
  * data/processed/phishing_dataset.csv
  * data/processed/dataset_preparation_report.txt

Tranco is deliberately NOT used: popularity does not prove a domain is
legitimate, so Tranco domains are never labelled or added to this dataset.

The cleaning, validation, deduplication and feature logic lives in
backend/app/ml/training/dataset_preparation.py and is unit-tested there.

Usage (from the project root):
    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import io
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.features import FEATURE_NAMES  # noqa: E402
from app.ml.training.dataset_preparation import (  # noqa: E402
    EXCLUDED_INVALID_URL,
    LABEL_NAMES,
    OUTPUT_COLUMNS,
    SOURCE_LEGITPHISH,
    SOURCE_OPENPHISH,
    PreparationResult,
    file_timestamp,
    load_legitphish,
    load_openphish,
    prepare_records,
    write_dataset,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OPENPHISH_FEED = RAW_DIR / "openphish_feed.txt"
LEGITPHISH_ZIP = RAW_DIR / "LegitPhish Dataset.zip"
LEGITPHISH_DIR = INTERIM_DIR / "legitphish"
DATASET_PATH = PROCESSED_DIR / "phishing_dataset.csv"
REPORT_PATH = PROCESSED_DIR / "dataset_preparation_report.txt"

MAX_LISTED = 10
MAX_CONFLICTS_LISTED = 50


def find_legitphish_csv() -> Path:
    candidates = sorted(LEGITPHISH_DIR.rglob("*.csv"))
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected exactly one CSV under {rel(LEGITPHISH_DIR)}, found {len(candidates)}. "
            "Run scripts/inspect_datasets.py first to extract it."
        )
    return candidates[0]


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "n/a"


def format_report(
    result: PreparationResult,
    legitphish_csv: Path,
    timestamps: dict[str, str],
) -> str:
    out: list[str] = []
    add = out.append
    rule = "=" * 78
    total_in = sum(result.input_counts.values())
    total_out = len(result.rows)

    add(rule)
    add("PhishGuard AI - Dataset Preparation Report")
    add(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    add("Generator: scripts/prepare_dataset.py")
    add("Label convention: 1 = phishing, 0 = legitimate")
    add(rule)

    add("\n1. SOURCES")
    add(f"  {SOURCE_OPENPHISH}:  {rel(OPENPHISH_FEED)}")
    add("     every URL -> label 1 (phishing)")
    add(f"     collected_at = {timestamps[SOURCE_OPENPHISH]} (feed file modification time)")
    add(f"  {SOURCE_LEGITPHISH}: {rel(legitphish_csv)}")
    add("     ClassLabel 0 (Phishing) -> label 1; ClassLabel 1 (Legitimate) -> label 0")
    add("     only the URL and ClassLabel columns are read; the 16 precomputed")
    add("     feature columns are ignored and all features are recomputed")
    add(f"     collected_at = {timestamps[SOURCE_LEGITPHISH]} ({rel(LEGITPHISH_ZIP)} modification time)")
    add("  collected_at records when this project obtained the source file, not")
    add("  when a URL was first observed online (the sources do not provide that).")
    add("  Tranco: NOT USED. It is kept separate as candidate-legitimate domains only")
    add("  and contributes no rows or labels to this dataset.")

    add("\n2. INPUT COUNTS")
    for source, count in sorted(result.input_counts.items()):
        add(f"  {source:<12} {count:>10,}")
    add(f"  {'total':<12} {total_in:>10,}")

    add("\n3. EXCLUDED ROWS")
    exclusion_counts = result.exclusion_counts()
    if not exclusion_counts:
        add("  none")
    for (source, reason), count in sorted(exclusion_counts.items()):
        add(f"  {source:<12} {reason:<22} {count:>8,}")
    add(f"  {'total':<35} {len(result.exclusions):>8,}")

    invalid_by_kind: dict[str, list] = defaultdict(list)
    for exclusion in result.exclusions:
        if exclusion.reason == EXCLUDED_INVALID_URL:
            kind = exclusion.detail.split(":", 1)[0]
            invalid_by_kind[kind].append(exclusion)
    if invalid_by_kind:
        add("  Invalid URL reasons:")
        for kind, items in sorted(invalid_by_kind.items(), key=lambda kv: -len(kv[1])):
            add(f"    {kind}: {len(items):,}")
            for exclusion in items[:MAX_LISTED]:
                add(f"      {exclusion.record.location}  {exclusion.record.raw_url[:110]!r}")
    for exclusion in result.exclusions:
        if exclusion.reason in {"missing_label", "unsupported_label", "empty_url"}:
            add(f"  {exclusion.reason}: {exclusion.record.location}  "
                f"url={exclusion.record.raw_url[:80]!r} label={exclusion.record.raw_label!r}")

    add("\n4. CLEANING ACTIONS (on rows that passed validation; a row can have several)")
    if not result.cleaning_action_counts:
        add("  none")
    for action, count in result.cleaning_action_counts.most_common():
        add(f"  {action:<32} {count:>8,}")
        for raw, cleaned in result.cleaning_examples.get(action, [])[:3]:
            add(f"      {raw[:70]!r}")
            add(f"        -> {cleaned[:70]}")
    add("  Not changed on purpose (would require guessing):")
    amp_rows = sum(1 for row in result.rows if "&amp;" in row.url)
    add(f"    HTML entity '&amp;' left as-is in {amp_rows:,} output URLs")
    add("    truncated schemes (e.g. 'ttp://') and inner whitespace -> excluded, not repaired")
    add("  Missing schemes: 'http://' is added, the same default the backend applies")
    add("  to scheme-less input at inference time, so has_https = 0 for those rows.")

    add("\n5. DUPLICATES (by normalized URL, identical labels)")
    add(f"  Exact duplicates (same cleaned string):          {result.exact_duplicates:,}")
    add(f"  Normalization-only duplicates (case, scheme...): {result.normalized_duplicates:,}")
    add(f"  URLs supplied by more than one source:           {result.cross_source_duplicates:,}")
    add("  The first occurrence is kept; 'source' lists every source that supplied it.")

    add("\n6. CONFLICTING LABELS (same normalized URL labelled both 0 and 1)")
    conflict_rows = sum(len(c.records) for c in result.conflicts)
    add(f"  Conflicting URLs: {len(result.conflicts):,} (all {conflict_rows:,} rows excluded)")
    for conflict in result.conflicts[:MAX_CONFLICTS_LISTED]:
        add(f"  {conflict.url}")
        for record in conflict.records:
            add(f"      {record.location}  source={record.source}  label={record.label}")
    if len(result.conflicts) > MAX_CONFLICTS_LISTED:
        add(f"  ... {len(result.conflicts) - MAX_CONFLICTS_LISTED:,} more not listed")

    add("\n7. OUTPUT DATASET")
    add(f"  File:    {rel(DATASET_PATH)}")
    add(f"  Columns ({len(OUTPUT_COLUMNS)}): {', '.join(OUTPUT_COLUMNS)}")
    add(f"  Rows:    {total_out:,}  (from {total_in:,} input rows)")
    add("  Class counts:")
    for label in sorted(LABEL_NAMES, reverse=True):
        count = result.class_counts[label]
        add(f"    {label} ({LABEL_NAMES[label]}): {count:>8,}  {pct(count, total_out)}")
    add("  Source counts:")
    for source, count in sorted(result.source_counts.items()):
        add(f"    {source:<22} {count:>8,}")
    add("  Class by source:")
    by_source_label = Counter((row.source, row.label) for row in result.rows)
    for (source, label), count in sorted(by_source_label.items()):
        add(f"    {source:<22} label {label}: {count:>8,}")

    add("\n8. NOTES FOR TRAINING (not acted on here)")
    hosts_by_label: dict[int, Counter] = {0: Counter(), 1: Counter()}
    for row in result.rows:
        hosts_by_label[row.label][row.domain] += 1
    for label in (1, 0):
        add(f"  Most frequent domains, label {label} (domain: rows):")
        for domain, count in hosts_by_label[label].most_common(5):
            add(f"    {domain}: {count:,}")
    add("  * Keep all rows of one domain in the same train/test split (group split)")
    add("    to avoid leakage from heavily repeated hosts.")
    add("  * Classes are imbalanced; use stratified splits and class-aware metrics.")
    add("  * Features are the 15 backend features:")
    add(f"    {', '.join(FEATURE_NAMES)}")

    add("\n" + rule)
    return "\n".join(out) + "\n"


def main() -> None:
    legitphish_csv = find_legitphish_csv()
    timestamps = {
        SOURCE_OPENPHISH: file_timestamp(OPENPHISH_FEED),
        SOURCE_LEGITPHISH: file_timestamp(LEGITPHISH_ZIP),
    }

    records = chain(
        load_openphish(OPENPHISH_FEED, timestamps[SOURCE_OPENPHISH]),
        load_legitphish(legitphish_csv, timestamps[SOURCE_LEGITPHISH]),
    )
    result = prepare_records(records)
    write_dataset(result.rows, DATASET_PATH)

    report = format_report(result, legitphish_csv, timestamps)
    REPORT_PATH.write_text(report, encoding="utf-8")

    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    stdout.write(report)
    stdout.write(f"Dataset written to {rel(DATASET_PATH)}\n")
    stdout.write(f"Report written to {rel(REPORT_PATH)}\n")
    stdout.flush()


if __name__ == "__main__":
    main()
