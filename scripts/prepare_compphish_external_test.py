"""Prepare a leakage-safe CompPhish external evaluation dataset.

Reads only the CompPhish mapping workbook and the existing project CSV. It
does not open HTML, fetch URLs, merge datasets, or train models.

Run from the project root:
    python scripts/prepare_compphish_external_test.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.features import (  # noqa: E402
    NormalizedURL,
    domain_group_key,
    extract_domain_info,
    extract_features_from_normalized,
)
from app.ml.features.normalization import InvalidURLError  # noqa: E402
from app.ml.training.dataset_preparation import (  # noqa: E402
    ACTION_ADDED_DEFAULT_SCHEME,
    ACTION_NORMALIZED_BY_BACKEND,
    OUTPUT_COLUMNS,
    clean_url,
    file_timestamp,
    validate_dataset_url,
)

MAPPING_FILE = PROJECT_ROOT / "data" / "interim" / "comp_phish" / "CompPhish" / "Mapping_File.xlsx"
EXISTING_DATASET = PROJECT_ROOT / "data" / "processed" / "phishing_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "compphish_external_test.csv"
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "compphish_external_test_report.txt"

SOURCE_COMPPHISH = "compphish"
OUTPUT_FIELDNAMES = (*OUTPUT_COLUMNS, "source_record_id")
EXCLUSION_REASONS = (
    "missing_label",
    "unsupported_label",
    "empty_url",
    "invalid_url",
    "registrable_domain_overlap",
    "conflicting_normalized_url_labels",
    "duplicate_normalized_url",
)

_NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass(frozen=True, slots=True)
class CompPhishRecord:
    row_number: int
    source_record_id: str
    raw_url: str
    raw_label: str


@dataclass(frozen=True, slots=True)
class Exclusion:
    record: CompPhishRecord
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Candidate:
    record: CompPhishRecord
    label: int
    parsed_url: NormalizedURL
    group_key: str
    cleaning_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExistingDomainGroups:
    row_count: int
    keys: frozenset[str]


@dataclass
class ExternalTestResult:
    input_count: int
    valid_record_count: int = 0
    rows: list[dict[str, str | int]] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    normalized_url_group_count: int = 0
    duplicate_normalized_url_group_count: int = 0
    duplicate_normalized_url_record_count: int = 0
    conflicting_normalized_url_group_count: int = 0
    conflicting_normalized_url_record_count: int = 0
    overlapping_url_group_count: int = 0
    overlapping_record_count: int = 0


def _column_index(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference)
    if match is None:
        raise ValueError(f"Invalid worksheet cell reference: {reference!r}")
    index = 0
    for character in match.group(0):
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def read_first_worksheet(path: Path) -> tuple[str, list[tuple[int, list[str]]]]:
    """Read the first worksheet's cached cell values using ZIP/XML only."""
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            strings_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{{{_NS['s']}}}t"))
                for item in strings_root.findall("s:si", _NS)
            ]

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships_root = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        sheets = workbook_root.find("s:sheets", _NS)
        if sheets is None or not len(sheets):
            raise ValueError(f"{path.name} has no worksheets")
        sheet = sheets[0]
        relationship_id = sheet.get(f"{{{_NS['r']}}}id")
        targets = {
            relation.get("Id"): relation.get("Target")
            for relation in relationships_root.findall("rel:Relationship", _NS)
        }
        target = targets.get(relationship_id)
        if not target:
            raise ValueError(f"{path.name} has no target for its first worksheet")

        target = target.replace("\\", "/")
        target = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
        if not target.startswith("xl/") or ".." in PurePosixPath(target).parts:
            raise ValueError(f"{path.name} has an unsafe worksheet target")
        if target not in workbook.namelist():
            raise ValueError(f"{path.name} is missing worksheet XML {target!r}")

        rows: list[tuple[int, list[str]]] = []
        with workbook.open(target) as worksheet:
            for _, element in ET.iterparse(worksheet, events=("end",)):
                if element.tag != f"{{{_NS['s']}}}row":
                    continue
                cells: dict[int, str] = {}
                for cell in element.findall("s:c", _NS):
                    cell_type = cell.get("t")
                    value_node = cell.find("s:v", _NS)
                    value = ""
                    if cell_type == "s" and value_node is not None:
                        value = shared_strings[int(value_node.text or "")]
                    elif cell_type == "inlineStr":
                        value = "".join(
                            node.text or ""
                            for node in cell.iter(f"{{{_NS['s']}}}t")
                        )
                    elif value_node is not None:
                        value = value_node.text or ""
                    cells[_column_index(cell.get("r") or "")] = value
                width = max(cells, default=-1) + 1
                row_values = [cells.get(index, "") for index in range(width)]
                try:
                    row_number = int(element.get("r", str(len(rows) + 1)))
                except ValueError:
                    row_number = len(rows) + 1
                rows.append((row_number, row_values))
                element.clear()

    return sheet.get("name") or "Sheet1", rows


def _normalize_source_record_id(value: str, row_number: int) -> str:
    text = value.strip()
    if not text:
        return f"row-{row_number}"
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if number.is_finite() and number == number.to_integral_value():
        return str(int(number))
    return text


def load_compphish_mapping(path: Path) -> tuple[str, list[CompPhishRecord]]:
    """Read only serial_number, url, and label from the mapping workbook."""
    sheet_name, worksheet_rows = read_first_worksheet(path)
    if not worksheet_rows:
        raise ValueError(f"{path.name} has no worksheet rows")

    header = [value.strip().casefold() for value in worksheet_rows[0][1]]
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name} has duplicate column names")
    columns = {name: index for index, name in enumerate(header)}
    required = {"serial_number", "url", "label"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    records: list[CompPhishRecord] = []
    for row_number, values in worksheet_rows[1:]:
        if not any(value.strip() for value in values):
            continue
        values += [""] * (len(header) - len(values))
        records.append(
            CompPhishRecord(
                row_number=row_number,
                source_record_id=_normalize_source_record_id(
                    values[columns["serial_number"]], row_number
                ),
                raw_url=values[columns["url"]],
                raw_label=values[columns["label"]],
            )
        )
    return sheet_name, records


def load_existing_domain_groups(path: Path) -> ExistingDomainGroups:
    """Recalculate all existing dataset grouping keys from its URL column."""
    keys: set[str] = set()
    row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "url" not in (reader.fieldnames or ()):
            raise ValueError(f"{path.name} is missing the url column")
        for row in reader:
            row_count += 1
            raw_url = (row.get("url") or "").strip()
            if not raw_url:
                raise ValueError(
                    f"{path.name} line {reader.line_num} has an empty URL; "
                    "cannot establish leakage groups safely"
                )
            try:
                parsed = validate_dataset_url(raw_url)
            except InvalidURLError as exc:
                raise ValueError(
                    f"{path.name} line {reader.line_num} has an invalid URL: {exc}"
                ) from exc
            keys.add(domain_group_key(parsed.hostname))
    return ExistingDomainGroups(row_count=row_count, keys=frozenset(keys))


def _parse_label(raw_label: str) -> tuple[int | None, str | None]:
    text = raw_label.strip()
    if not text:
        return None, "missing_label"
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None, "unsupported_label"
    if not value.is_finite() or value not in (Decimal(0), Decimal(1)):
        return None, "unsupported_label"
    return int(value), None


def _process_candidate(record: CompPhishRecord) -> tuple[_Candidate | None, Exclusion | None]:
    label, label_error = _parse_label(record.raw_label)
    if label_error:
        return None, Exclusion(record, label_error, record.raw_label)
    if label is None:
        raise ValueError("Label parser returned no label without an exclusion reason")

    cleaned = clean_url(record.raw_url)
    if not cleaned.url:
        return None, Exclusion(record, "empty_url")

    try:
        parsed = validate_dataset_url(cleaned.url)
    except InvalidURLError as exc:
        return None, Exclusion(record, "invalid_url", str(exc))

    actions = list(cleaned.actions)
    if not parsed.scheme_was_explicit:
        actions.append(ACTION_ADDED_DEFAULT_SCHEME)
    elif parsed.url != cleaned.url:
        actions.append(ACTION_NORMALIZED_BY_BACKEND)

    return (
        _Candidate(
            record=record,
            label=label,
            parsed_url=parsed,
            group_key=domain_group_key(parsed.hostname),
            cleaning_actions=tuple(actions),
        ),
        None,
    )


def _output_row(candidate: _Candidate, collected_at: str) -> dict[str, str | int]:
    features = extract_features_from_normalized(candidate.parsed_url).to_dict()
    return {
        "url": candidate.parsed_url.url,
        "domain": candidate.parsed_url.hostname,
        "label": candidate.label,
        "source": SOURCE_COMPPHISH,
        "collected_at": collected_at,
        **features,
        "cleaning_actions": ";".join(candidate.cleaning_actions),
        "source_record_id": candidate.record.source_record_id,
    }


def prepare_external_test(
    records: list[CompPhishRecord],
    existing_group_keys: frozenset[str] | set[str],
    collected_at: str,
) -> ExternalTestResult:
    """Validate, normalize, de-leak, deduplicate, and featurize CompPhish rows.

    Domain overlap has exclusion priority. Duplicate and conflicting normalized
    URL groups are still counted for audit, including groups already excluded
    for overlap.
    """
    result = ExternalTestResult(input_count=len(records))
    groups: dict[str, list[_Candidate]] = defaultdict(list)

    for record in records:
        candidate, exclusion = _process_candidate(record)
        if exclusion is not None:
            result.exclusions.append(exclusion)
            continue
        assert candidate is not None
        result.valid_record_count += 1
        groups[candidate.parsed_url.url].append(candidate)

    result.normalized_url_group_count = len(groups)
    for normalized_url, candidates in groups.items():
        labels = {candidate.label for candidate in candidates}
        if len(candidates) > 1:
            result.duplicate_normalized_url_group_count += 1
            result.duplicate_normalized_url_record_count += len(candidates) - 1
        if len(labels) > 1:
            result.conflicting_normalized_url_group_count += 1
            result.conflicting_normalized_url_record_count += len(candidates)

        group_key = candidates[0].group_key
        if group_key in existing_group_keys:
            result.overlapping_url_group_count += 1
            result.overlapping_record_count += len(candidates)
            detail = f"group_key={group_key}"
            if len(labels) > 1:
                detail += "; also has conflicting labels"
            elif len(candidates) > 1:
                detail += "; also has duplicate normalized URLs"
            result.exclusions.extend(
                Exclusion(candidate.record, "registrable_domain_overlap", detail)
                for candidate in candidates
            )
            continue

        if len(labels) > 1:
            label_text = ",".join(str(label) for label in sorted(labels))
            result.exclusions.extend(
                Exclusion(
                    candidate.record,
                    "conflicting_normalized_url_labels",
                    f"url={normalized_url}; labels={label_text}",
                )
                for candidate in candidates
            )
            continue

        result.rows.append(_output_row(candidates[0], collected_at))
        for duplicate in candidates[1:]:
            result.exclusions.append(
                Exclusion(
                    duplicate.record,
                    "duplicate_normalized_url",
                    f"url={normalized_url}; kept={candidates[0].record.source_record_id}",
                )
            )

    result.exclusions.sort(key=lambda exclusion: exclusion.record.row_number)
    return result


def write_external_test(
    rows: list[dict[str, str | int]], path: Path
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentage(part: int, total: int) -> str:
    return f"{100 * part / total:.2f}%" if total else "n/a"


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def format_report(
    result: ExternalTestResult,
    existing: ExistingDomainGroups,
    mapping_file: Path,
    existing_file: Path,
    sheet_name: str,
    collected_at: str,
) -> str:
    out: list[str] = []
    add = out.append
    rule = "=" * 88
    exclusion_counts = Counter(exclusion.reason for exclusion in result.exclusions)
    retained = len(result.rows)

    add(rule)
    add("PhishGuard AI - CompPhish External Evaluation Dataset Report")
    add("Generator: scripts/prepare_compphish_external_test.py")
    add("Label convention: 1 = phishing, 0 = legitimate (CompPhish labels used as provided)")
    add("Status: External evaluation set; not a verified ground-truth dataset.")
    add(rule)

    add("\n1. INPUTS AND REPRODUCIBILITY")
    add(f"  CompPhish mapping: {_relative_path(mapping_file)}")
    add(f"  Worksheet: {sheet_name}")
    add("  Mapping columns used: serial_number, url, label")
    add("  CompPhish precomputed features: ignored; all 15 features recalculated")
    add(f"  Existing dataset: {_relative_path(existing_file)}")
    add("  Existing dataset column used for grouping: url only; stored domain values ignored")
    add(f"  CompPhish mapping SHA-256: {_sha256(mapping_file)}")
    add(f"  Existing dataset SHA-256: {_sha256(existing_file)}")
    add(f"  collected_at for retained records: {collected_at} (mapping workbook file mtime)")
    add("  This report omits a run timestamp so identical inputs and file metadata produce")
    add("  deterministic output rows and report contents.")

    add("\n2. DOMAIN LEAKAGE POLICY")
    add("  Both datasets are URL-normalized with the project validator, then grouped using")
    add("  the pinned PSL utility, including private suffixes. Stored domain columns are not")
    add("  trusted. A host without an eTLD+1 uses its normalized hostname as the group key.")
    add(f"  Existing rows: {existing.row_count:,}; unique existing group keys: {len(existing.keys):,}")
    add("  Any CompPhish URL group with an existing group key is excluded before evaluation.")

    add("\n3. INPUT AND PROCESSING COUNTS")
    add(f"  CompPhish mapping records: {result.input_count:,}")
    add(f"  Records with a supported label and valid normalized URL: {result.valid_record_count:,}")
    add(f"  Unique normalized URL groups among valid records: {result.normalized_url_group_count:,}")
    add(f"  Repeated normalized URL groups: {result.duplicate_normalized_url_group_count:,}")
    add(f"  Extra records in repeated normalized URL groups: {result.duplicate_normalized_url_record_count:,}")
    add(f"  Conflicting-label normalized URL groups: {result.conflicting_normalized_url_group_count:,}")
    add(f"  Records in conflicting-label normalized URL groups: {result.conflicting_normalized_url_record_count:,}")
    add(f"  Overlapping normalized URL groups excluded: {result.overlapping_url_group_count:,}")
    add(f"  Records excluded for existing domain/group overlap: {result.overlapping_record_count:,}")
    add(f"  Retained external evaluation records: {retained:,}")
    add(f"  Excluded records total: {len(result.exclusions):,}")
    add("  Excluded counts by primary reason:")
    for reason in EXCLUSION_REASONS:
        add(f"    {reason:<38} {exclusion_counts[reason]:>8,}")
    if retained + len(result.exclusions) != result.input_count:
        raise ValueError("Internal count mismatch: input rows are not fully accounted for")

    add("\n4. RETAINED CLASS AND URL DISTRIBUTIONS")
    class_counts = Counter(int(row["label"]) for row in result.rows)
    for label, name in ((1, "phishing"), (0, "legitimate")):
        count = class_counts[label]
        add(f"  Label {label} ({name}): {count:,} ({_percentage(count, retained)})")

    https_counts = Counter(int(row["has_https"]) for row in result.rows)
    add("  HTTPS distribution:")
    add(f"    HTTPS: {https_counts[1]:,} ({_percentage(https_counts[1], retained)})")
    add(f"    HTTP:  {https_counts[0]:,} ({_percentage(https_counts[0], retained)})")

    shapes: Counter[str] = Counter()
    for row in result.rows:
        parts = urlsplit(str(row["url"]))
        if parts.path in ("", "/"):
            shapes["root + query" if parts.query else "homepage"] += 1
        else:
            shapes["path"] += 1
    add("  URL shape (homepage = root with no query; path = non-root path):")
    for shape in ("homepage", "root + query", "path"):
        count = shapes[shape]
        add(f"    {shape:<14} {count:>8,} ({_percentage(count, retained)})")

    registrable_domains: set[str] = set()
    fallback_group_keys: set[str] = set()
    all_group_keys: set[str] = set()
    for row in result.rows:
        hostname = str(row["domain"])
        info = extract_domain_info(hostname)
        key = domain_group_key(hostname)
        all_group_keys.add(key)
        if info.registrable_domain:
            registrable_domains.add(info.registrable_domain)
        else:
            fallback_group_keys.add(key)
    add(f"  Unique registrable domains (eTLD+1): {len(registrable_domains):,}")
    add(f"  Unique fallback host groups (no eTLD+1): {len(fallback_group_keys):,}")
    add(f"  Unique total group keys: {len(all_group_keys):,}")

    add("\n5. EXCLUSIONS (one primary reason per input record)")
    add("  Domain overlap has priority over duplicate/conflicting URL reasons; such URL")
    add("  groups are still included in the duplicate/conflict audit counts above.")
    for exclusion in result.exclusions:
        record = exclusion.record
        add(
            f"  workbook_row={record.row_number} "
            f"source_record_id={record.source_record_id!r} "
            f"reason={exclusion.reason} raw_label={record.raw_label!r} "
            f"raw_url={record.raw_url!r} detail={exclusion.detail!r}"
        )

    add("\n6. LIMITATIONS")
    add("  This is an external evaluation set with registrable-domain overlap removed; it")
    add("  is not a verified ground-truth dataset. Source labels are retained as provided")
    add("  and are not independently validated or silently relabelled.")
    add("  No CompPhish HTML was opened or executed. URLs were not fetched, resolved, or")
    add("  checked against live services. Redirect destinations and page contents were not")
    add("  used. Domain disjointness does not rule out temporal, campaign, or source bias.")
    add("  CompPhish collection dates are not available per row; collected_at records the")
    add("  mapping workbook file mtime, not when each URL was observed.")
    add("\n" + rule)
    return "\n".join(out) + "\n"


def main() -> None:
    sheet_name, records = load_compphish_mapping(MAPPING_FILE)
    existing = load_existing_domain_groups(EXISTING_DATASET)
    collected_at = file_timestamp(MAPPING_FILE)
    result = prepare_external_test(records, existing.keys, collected_at)
    report = format_report(
        result,
        existing,
        MAPPING_FILE,
        EXISTING_DATASET,
        sheet_name,
        collected_at,
    )

    write_external_test(result.rows, OUTPUT_FILE)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(
        f"Retained {len(result.rows):,} of {result.input_count:,} CompPhish records; "
        f"excluded {len(result.exclusions):,}."
    )
    print(f"Dataset: {OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Report: {REPORT_FILE.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
