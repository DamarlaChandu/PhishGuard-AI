"""Read-only inspection of the CompPhish dataset.

What it does:
  * Lists data/raw/CompPhish.zip and its nested archives before extracting.
  * Safely extracts the outer archive into data/interim/comp_phish/ (path
    traversal and size checks). The nested All_HTML.zip (~3.5 GB uncompressed)
    is NOT extracted; its members are streamed and only the first bytes of each
    file are read. Nested code archives are listed and read as text only;
    nothing from the dataset is executed or unpickled.
  * Reads the README and data dictionary with standard-library XML parsing.
  * Profiles Mapping_File.xlsx (URL + label), verifies the label direction,
    compares with data/processed/phishing_dataset.csv, and checks HTML coverage.
  * Writes data/processed/compphish_inspection_report.txt.

It never fetches URLs, merges datasets, trains models or modifies inputs.

Usage (from the project root):
    python scripts/inspect_compphish.py
"""

from __future__ import annotations

import csv
import io
import re
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.ml.features import InvalidURLError, extract_features_from_normalized  # noqa: E402
from app.ml.features.extractor import count_subdomains  # noqa: E402
from app.ml.features.ip_address import parse_ip_hostname  # noqa: E402
from app.ml.training.dataset_preparation import validate_dataset_url  # noqa: E402
from inspect_datasets import safe_extract  # noqa: E402

RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "CompPhish.zip"
EXTRACT_DIR = PROJECT_ROOT / "data" / "interim" / "comp_phish"
EXISTING_DATASET = PROJECT_ROOT / "data" / "processed" / "phishing_dataset.csv"
TRANCO_CSV = PROJECT_ROOT / "data" / "interim" / "tranco" / "top-1m.csv"
OPENPHISH_FEED = PROJECT_ROOT / "data" / "raw" / "openphish_feed.txt"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "compphish_inspection_report.txt"

# Nested archives above this uncompressed size are streamed, not extracted.
NESTED_EXTRACT_LIMIT_BYTES = 200 * 1024**2
HTML_SNIFF_BYTES = 4096
TINY_HTML_BYTES = 512
SAMPLE = 5

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


# --------------------------------------------------------------------------- #
# Office documents (stdlib only; openpyxl / python-docx are not installed)
# --------------------------------------------------------------------------- #
def docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    return [
        "".join(t.text or "" for t in p.iter(f"{{{NS['w']}}}t"))
        for p in root.iter(f"{{{NS['w']}}}p")
    ]


def _column_index(ref: str) -> int:
    index = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        index = index * 26 + ord(ch) - 64
    return index - 1


def xlsx_first_sheet(path: Path) -> tuple[str, list[list[str]]]:
    """Return (sheet name, rows) of the first worksheet as strings."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.iter(f"{{{NS['s']}}}t"))
                for si in root.findall("s:si", NS)
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels.findall("rel:Relationship", NS)}
        sheet = workbook.find("s:sheets", NS)[0]
        target = targets[sheet.get(f"{{{NS['r']}}}id")].lstrip("/")
        target = target if target.startswith("xl/") else f"xl/{target}"

        rows: list[list[str]] = []
        with archive.open(target) as handle:
            for _, elem in ET.iterparse(handle):
                if elem.tag != f"{{{NS['s']}}}row":
                    continue
                cells: dict[int, str] = {}
                for cell in elem.findall("s:c", NS):
                    kind, value_node = cell.get("t"), cell.find("s:v", NS)
                    if kind == "s" and value_node is not None:
                        value = shared[int(value_node.text)]
                    elif kind == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(f"{{{NS['s']}}}t"))
                    else:
                        value = value_node.text if value_node is not None else ""
                    cells[_column_index(cell.get("r"))] = value
                width = max(cells) + 1 if cells else 0
                rows.append([cells.get(i, "") for i in range(width)])
                elem.clear()
    return sheet.get("name"), rows


def rows_to_dicts(rows: list[list[str]]) -> tuple[list[str], list[dict[str, str]]]:
    header = [h.strip() for h in rows[0]]
    records = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * (len(header) - len(row))
        records.append({header[i]: padded[i] for i in range(len(header))})
    return header, records


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def registrable_domain(hostname: str) -> str:
    if parse_ip_hostname(hostname) is not None:
        return hostname
    labels = hostname.split(".")
    return ".".join(labels[count_subdomains(hostname):])


def normalize_serial(value: str) -> str | None:
    """Excel stores numbers as text like "12" or "12.0"; return "12"."""
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() and number >= 0 else None


def url_shape(path: str, query: str) -> str:
    if path in ("", "/"):
        return "homepage" if not query else "root + query"
    return "path"


def auc(scores_positive: list[float], scores_negative: list[float]) -> float | None:
    """Mann-Whitney ROC-AUC (ties count half); stdlib only."""
    if not scores_positive or not scores_negative:
        return None
    combined = sorted(
        [(s, 1) for s in scores_positive] + [(s, 0) for s in scores_negative], key=lambda x: x[0]
    )
    rank_sum, i = 0.0, 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        rank_sum += average_rank * sum(1 for k in range(i, j) if combined[k][1] == 1)
        i = j
    n_pos, n_neg = len(scores_positive), len(scores_negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "n/a"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


# --------------------------------------------------------------------------- #
# Archive inspection
# --------------------------------------------------------------------------- #
@dataclass
class ArchiveSummary:
    name: str
    members: int
    files: int
    uncompressed: int
    compressed: int
    encrypted: int
    symlinks: int
    unsafe_paths: int
    max_ratio: float
    extensions: Counter
    largest: list[tuple[str, int]]


def summarize_archive(archive: zipfile.ZipFile, name: str) -> ArchiveSummary:
    infos = archive.infolist()

    def is_unsafe(member: str) -> bool:
        normalized = member.replace("\\", "/")
        return normalized.startswith("/") or ":" in normalized or ".." in normalized.split("/")

    def extension(info: zipfile.ZipInfo) -> str:
        base = info.filename.rsplit("/", 1)[-1]
        return base.rsplit(".", 1)[-1].lower() if "." in base else "<none>"

    files = [i for i in infos if not i.is_dir()]
    return ArchiveSummary(
        name=name,
        members=len(infos),
        files=len(files),
        uncompressed=sum(i.file_size for i in infos),
        compressed=sum(i.compress_size for i in infos),
        encrypted=sum(i.flag_bits & 1 for i in infos),
        symlinks=sum(((i.external_attr >> 16) & 0o170000) == 0o120000 for i in infos),
        unsafe_paths=sum(is_unsafe(i.filename) for i in infos),
        max_ratio=max((i.file_size / max(i.compress_size, 1) for i in files), default=0.0),
        extensions=Counter(extension(i) for i in files),
        largest=[(i.filename, i.file_size) for i in sorted(files, key=lambda i: -i.file_size)[:3]],
    )


# --------------------------------------------------------------------------- #
# URL / label profiling
# --------------------------------------------------------------------------- #
@dataclass
class URLRecord:
    serial: str
    raw_url: str
    label: int
    url: str  # normalized
    hostname: str
    group: str
    scheme: str
    shape: str
    scheme_was_explicit: bool
    features: dict[str, int]


@dataclass
class MappingProfile:
    header: list[str]
    total_rows: int = 0
    raw_labels: Counter = field(default_factory=Counter)
    missing_labels: int = 0
    unsupported_labels: Counter = field(default_factory=Counter)
    missing_serials: int = 0
    duplicate_serials: int = 0
    empty_urls: int = 0
    invalid_urls: list[tuple[str, str, str]] = field(default_factory=list)  # serial, url, reason
    valid: list[URLRecord] = field(default_factory=list)
    exact_duplicate_rows: int = 0
    normalized_duplicate_rows: int = 0
    conflicting_urls: list[str] = field(default_factory=list)


def profile_mapping(header: list[str], records: list[dict[str, str]]) -> MappingProfile:
    profile = MappingProfile(header=header)
    serial_counts: Counter = Counter()
    raw_seen: Counter = Counter()
    labels_by_url: dict[str, set[int]] = {}
    normalized_seen: Counter = Counter()

    for row in records:
        profile.total_rows += 1
        serial = normalize_serial(row.get("serial_number", ""))
        if serial is None:
            profile.missing_serials += 1
        else:
            serial_counts[serial] += 1

        raw_label = row.get("label", "").strip()
        profile.raw_labels[raw_label] += 1
        label = {"0": 0, "1": 1, "0.0": 0, "1.0": 1}.get(raw_label)
        if not raw_label:
            profile.missing_labels += 1
            continue
        if label is None:
            profile.unsupported_labels[raw_label] += 1
            continue

        raw_url = row.get("url", "")
        if not raw_url.strip():
            profile.empty_urls += 1
            continue
        try:
            parsed = validate_dataset_url(raw_url)
        except InvalidURLError as exc:
            profile.invalid_urls.append((serial or "?", raw_url, str(exc)))
            continue

        raw_seen[raw_url.strip()] += 1
        normalized_seen[parsed.url] += 1
        labels_by_url.setdefault(parsed.url, set()).add(label)
        profile.valid.append(
            URLRecord(
                serial=serial or "?",
                raw_url=raw_url,
                label=label,
                url=parsed.url,
                hostname=parsed.hostname,
                group=registrable_domain(parsed.hostname),
                scheme=parsed.scheme,
                shape=url_shape(parsed.path, parsed.query),
                scheme_was_explicit=parsed.scheme_was_explicit,
                features=extract_features_from_normalized(parsed).to_dict(),
            )
        )

    profile.duplicate_serials = sum(c - 1 for c in serial_counts.values() if c > 1)
    profile.exact_duplicate_rows = sum(c - 1 for c in raw_seen.values() if c > 1)
    total_normalized_dups = sum(c - 1 for c in normalized_seen.values() if c > 1)
    profile.normalized_duplicate_rows = total_normalized_dups - profile.exact_duplicate_rows
    profile.conflicting_urls = sorted(u for u, labels in labels_by_url.items() if len(labels) > 1)
    return profile


# --------------------------------------------------------------------------- #
# External references (read-only)
# --------------------------------------------------------------------------- #
@dataclass
class ExistingDataset:
    url_labels: dict[str, set[int]]
    host_labels: dict[str, set[int]]
    group_labels: dict[str, set[int]]
    rows: int
    labels: Counter
    https_by_label: Counter
    shape_by_label: Counter


def load_existing_dataset(path: Path) -> ExistingDataset:
    url_labels: dict[str, set[int]] = {}
    host_labels: dict[str, set[int]] = {}
    group_labels: dict[str, set[int]] = {}
    labels: Counter = Counter()
    https_by_label: Counter = Counter()
    shape_by_label: Counter = Counter()
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            label = int(row["label"])
            labels[label] += 1
            url_labels.setdefault(row["url"], set()).add(label)
            host_labels.setdefault(row["domain"], set()).add(label)
            group_labels.setdefault(registrable_domain(row["domain"]), set()).add(label)
            https_by_label[(label, row["has_https"] == "1")] += 1
            path_length = int(row["path_length"])
            has_query = int(row["num_parameters"]) > 0 or "?" in row["url"]
            shape = "path" if path_length > 1 else ("root + query" if has_query else "homepage")
            shape_by_label[(label, shape)] += 1
    return ExistingDataset(url_labels, host_labels, group_labels, rows, labels, https_by_label, shape_by_label)


def load_tranco_domains(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {line.split(",", 1)[1].strip().lower() for line in handle if "," in line}


def load_openphish_urls(path: Path) -> set[str]:
    urls = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            urls.add(validate_dataset_url(line).url)
        except InvalidURLError:
            continue
    return urls


# --------------------------------------------------------------------------- #
# HTML coverage (streamed from the nested archive)
# --------------------------------------------------------------------------- #
@dataclass
class HTMLCoverage:
    members: int
    named_by_serial: int
    other_names: list[str]
    matched: int
    mapping_without_html: list[str]
    html_without_mapping: int
    tiny_files_by_label: Counter
    html_like_by_label: Counter
    sizes_by_label: dict[int, list[int]]


_HTML_MARKER = re.compile(rb"<\s*(!doctype|html|head|body|script|meta|div|title)", re.IGNORECASE)


def inspect_html(html_zip: Path, serial_labels: dict[str, int]) -> HTMLCoverage:
    named: dict[str, zipfile.ZipInfo] = {}
    other: list[str] = []
    with zipfile.ZipFile(html_zip) as archive:
        files = [i for i in archive.infolist() if not i.is_dir()]
        for info in files:
            stem = info.filename.rsplit("/", 1)[-1]
            match = re.fullmatch(r"(\d+)\.txt", stem)
            if match:
                named[str(int(match.group(1)))] = info
            else:
                other.append(info.filename)

        tiny: Counter = Counter()
        html_like: Counter = Counter()
        sizes: dict[int, list[int]] = {0: [], 1: []}
        for serial, label in serial_labels.items():
            info = named.get(serial)
            if info is None:
                continue
            sizes[label].append(info.file_size)
            if info.file_size < TINY_HTML_BYTES:
                tiny[label] += 1
            with archive.open(info) as handle:
                head = handle.read(HTML_SNIFF_BYTES)
            if _HTML_MARKER.search(head):
                html_like[label] += 1

    matched = sum(1 for s in serial_labels if s in named)
    return HTMLCoverage(
        members=len(files),
        named_by_serial=len(named),
        other_names=other,
        matched=matched,
        mapping_without_html=sorted((s for s in serial_labels if s not in named), key=int),
        html_without_mapping=sum(1 for s in named if s not in serial_labels),
        tiny_files_by_label=tiny,
        html_like_by_label=html_like,
        sizes_by_label=sizes,
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def main() -> None:
    out: list[str] = []
    add = out.append
    rule = "=" * 78
    add(rule)
    add("PhishGuard AI - CompPhish Dataset Inspection Report")
    add(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    add("Generator: scripts/inspect_compphish.py (read-only; no URLs fetched,")
    add("           no dataset code executed, nothing merged or trained)")
    add(rule)

    # 1. Archives -----------------------------------------------------------
    with zipfile.ZipFile(RAW_ZIP) as archive:
        corrupt = archive.testzip()
        outer = summarize_archive(archive, rel(RAW_ZIP))
    if corrupt is not None:
        raise SystemExit(f"CRC check failed for {corrupt!r}")
    extracted = safe_extract(RAW_ZIP, EXTRACT_DIR)
    base = EXTRACT_DIR / "CompPhish"

    nested = {}
    for name in ("All_HTML.zip", "dataset Creator.zip", "feature extractor.zip"):
        with zipfile.ZipFile(base / name) as archive:
            nested[name] = summarize_archive(archive, name)

    add("\n1. ARCHIVE CONTENTS AND EXTRACTION")
    for summary in (outer, *nested.values()):
        add(f"  {summary.name}")
        add(f"    files {summary.files:,} | {summary.uncompressed:,} bytes uncompressed "
            f"({summary.compressed:,} compressed) | max ratio {summary.max_ratio:.1f}x")
        add(f"    encrypted {summary.encrypted} | symlinks {summary.symlinks} | "
            f"unsafe paths {summary.unsafe_paths} | types {dict(summary.extensions)}")
    add(f"  Outer archive CRC check: OK. Extracted to {rel(EXTRACT_DIR)}/ "
        "(path-traversal and size checks passed):")
    for path in extracted.extracted_files:
        add(f"    {rel(path)}  ({path.stat().st_size:,} bytes)")
    html_summary = nested["All_HTML.zip"]
    add(f"  All_HTML.zip NOT extracted: {html_summary.uncompressed / 1024**3:.2f} GiB uncompressed "
        f"exceeds the {NESTED_EXTRACT_LIMIT_BYTES // 1024**2} MB nested-extraction limit;")
    add(f"  members are streamed and only the first {HTML_SNIFF_BYTES} bytes of each are read.")
    add("  Code archives (dataset Creator, feature extractor) were read as text only.")
    add("  feature extractor/gib_model.pki is a pickle and was NOT loaded (unsafe to unpickle).")

    # 2. Documentation ------------------------------------------------------
    readme = [p.strip() for p in docx_paragraphs(base / "Readme.docx") if p.strip()]
    add("\n2. DOCUMENTATION")
    add("  Readme.docx (key statements, quoted):")
    for paragraph in readme:
        lowered = paragraph.lower()
        if any(k in lowered for k in ("collection period", "label (", "not sanitised", "mapping_file.xlsx-", "all_html.zip –")):
            add(f"    \"{paragraph[:300]}\"")
    _, dictionary_rows = xlsx_first_sheet(base / "Data_Dictionary.xlsx")
    feature_rows = [r for r in dictionary_rows[1:] if r and r[0].strip().startswith("F")]
    categories = Counter(r[3].strip() for r in feature_rows if len(r) > 3)
    add(f"  Data_Dictionary.xlsx: {len(feature_rows)} features documented: {dict(categories)}")
    add("  The dictionary describes features only; it does not restate the label mapping.")
    add("  Dataset-creator code (read as text): bare domains were turned into URLs by")
    add("  requesting https:// first, then http://, keeping the final URL after redirects;")
    add("  Database_Cleaner.py keeps only rows whose HTML was downloaded (live sites only).")

    # 3. URL / label files --------------------------------------------------
    mapping_sheet, mapping_rows = xlsx_first_sheet(base / "Mapping_File.xlsx")
    header, mapping_records = rows_to_dicts(mapping_rows)
    features_sheet, feature_rows_all = xlsx_first_sheet(base / "All_Features_threshold90.xlsx")
    feature_header, feature_records = rows_to_dicts(feature_rows_all)
    profile = profile_mapping(header, mapping_records)

    add("\n3. URL AND LABEL FILES")
    add(f"  Mapping_File.xlsx (sheet {mapping_sheet!r}): columns {header}")
    add(f"  All_Features_threshold90.xlsx (sheet {features_sheet!r}): {len(feature_records):,} rows, "
        f"{len(feature_header)} columns (serial_number, url, label + "
        f"{len(feature_header) - 3} precomputed features; not used here)")
    mapping_key = {(normalize_serial(r['serial_number']), r['url'].strip(), r['label'].strip()) for r in mapping_records}
    feature_key = {(normalize_serial(r['serial_number']), r['url'].strip(), r['label'].strip()) for r in feature_records}
    add(f"  Rows identical (serial, url, label) in both files: {len(mapping_key & feature_key):,} | "
        f"only in mapping: {len(mapping_key - feature_key):,} | only in features: {len(feature_key - mapping_key):,}")

    valid = profile.valid
    by_label = Counter(r.label for r in valid)
    add("\n4. RECORD COUNTS AND DATA QUALITY (Mapping_File.xlsx)")
    add(f"  Total records:             {profile.total_rows:,}")
    add(f"  Raw label values:          {dict(profile.raw_labels)}")
    add(f"  Missing labels:            {profile.missing_labels:,}")
    add(f"  Unsupported labels:        {dict(profile.unsupported_labels) or 'none'}")
    add(f"  Missing/invalid serials:   {profile.missing_serials:,} | duplicate serials: {profile.duplicate_serials:,}")
    add(f"  Empty URLs:                {profile.empty_urls:,}")
    add(f"  Invalid URLs:              {len(profile.invalid_urls):,} (backend normalize_url + dotted-host rule)")
    for serial, url, reason in profile.invalid_urls[:SAMPLE * 2]:
        add(f"    serial {serial}: {url[:90]!r} -> {reason}")
    add(f"  URLs without explicit scheme: {sum(not r.scheme_was_explicit for r in valid):,}")
    add(f"  Duplicate URLs: {profile.exact_duplicate_rows:,} exact extra rows, "
        f"{profile.normalized_duplicate_rows:,} more after normalization")
    add(f"  Same URL with conflicting labels: {len(profile.conflicting_urls):,}")
    for url in profile.conflicting_urls[:SAMPLE]:
        add(f"    {url}")
    add(f"  Valid labelled URLs: {len(valid):,} | label 1: {by_label[1]:,} ({pct(by_label[1], len(valid))}) | "
        f"label 0: {by_label[0]:,} ({pct(by_label[0], len(valid))})")
    for label in (1, 0):
        add(f"  Sample label {label}:")
        for record in [r for r in valid if r.label == label][:SAMPLE]:
            add(f"    {record.url[:100]}")

    # 5. Label verification -------------------------------------------------
    tranco = load_tranco_domains(TRANCO_CSV)
    openphish = load_openphish_urls(OPENPHISH_FEED)
    existing = load_existing_dataset(EXISTING_DATASET)

    add("\n5. LABEL MAPPING VERIFICATION")
    add("  Documentation: Readme.docx states label 1 = phishing, 0 = legitimate.")
    add("  Data checks (valid URLs; popularity compared by registrable domain):")
    add(f"  {'':46}{'label 1':>12}{'label 0':>12}")
    for title, predicate in [
        ("distinct registrable domains", None),
        ("  in Tranco top 1M (popular)", "tranco"),
    ]:
        values = []
        for label in (1, 0):
            groups = {r.group for r in valid if r.label == label}
            values.append(len(groups) if predicate is None else sum(1 for g in groups if g in tranco))
        add(f"  {title:<46}{values[0]:>12,}{values[1]:>12,}")
    rates = []
    for label in (1, 0):
        groups = {r.group for r in valid if r.label == label}
        rates.append(pct(sum(1 for g in groups if g in tranco), len(groups)))
    add(f"  {'  as share of distinct domains':<46}{rates[0]:>12}{rates[1]:>12}")
    kw = [pct(sum(r.features['has_suspicious_keywords'] for r in valid if r.label == l), by_label[l]) for l in (1, 0)]
    add(f"  {'URLs with a suspicious keyword':<46}{kw[0]:>12}{kw[1]:>12}")
    op = [sum(1 for r in valid if r.label == l and r.url in openphish) for l in (1, 0)]
    add(f"  {'exact URL in OpenPhish feed (phishing)':<46}{op[0]:>12,}{op[1]:>12,}")
    agree = Counter()
    for r in valid:
        ours = existing.url_labels.get(r.url)
        if ours and len(ours) == 1:
            agree[(r.label, next(iter(ours)))] += 1
    add("  Overlapping URLs with our dataset, CompPhish label vs our label:")
    for (theirs, ours), count in sorted(agree.items()):
        add(f"    CompPhish {theirs} / ours {ours}: {count:,}")
    if not agree:
        add("    no overlapping URLs")
    tranco_rate = {l: (sum(1 for g in {r.group for r in valid if r.label == l} if g in tranco),
                       len({r.group for r in valid if r.label == l})) for l in (0, 1)}
    r0 = tranco_rate[0][0] / tranco_rate[0][1] if tranco_rate[0][1] else 0
    r1 = tranco_rate[1][0] / tranco_rate[1][1] if tranco_rate[1][1] else 0
    consistent = r0 > r1 and op[0] >= op[1]
    add("  Verdict: " + (
        "CONSISTENT with documentation (1 = phishing, 0 = legitimate): label-0 domains are "
        f"far more often popular (Tranco {r0:.1%} vs {r1:.1%})."
        if consistent else
        "NOT CONFIRMED by the data checks; review before use."
    ))

    # 6. HTTPS and URL shape ------------------------------------------------
    add("\n6. HTTPS vs HTTP AND HOMEPAGE vs PATH, PER CLASS")
    add("  'homepage' = empty or '/' path and no query; 'root + query' = '/' with a query;")
    add("  'path' = anything after the host.")
    for label, name in ((1, "phishing"), (0, "legitimate")):
        records = [r for r in valid if r.label == label]
        n = len(records)
        https = sum(r.scheme == "https" for r in records)
        shapes = Counter(r.shape for r in records)
        add(f"  CompPhish {name} (n={n:,}): https {pct(https, n)} | http {pct(n - https, n)}")
        add(f"    homepage {pct(shapes['homepage'], n)} | root + query {pct(shapes['root + query'], n)} | path {pct(shapes['path'], n)}")
    add("  Cross-tab (label, scheme, shape) counts:")
    cross = Counter((r.label, r.scheme, r.shape) for r in valid)
    for key, count in sorted(cross.items()):
        add(f"    label {key[0]} | {key[1]:<5} | {key[2]:<12} {count:>7,}")

    add("  Single-feature separation on CompPhish (ROC-AUC of the raw feature, 1 = phishing):")
    for feature in ("has_https", "path_length", "num_digits", "url_length", "num_dots", "num_subdomains"):
        score = auc([r.features[feature] for r in valid if r.label == 1],
                    [r.features[feature] for r in valid if r.label == 0])
        separation = max(score, 1 - score) if score is not None else None
        add(f"    {feature:<18} AUC {score:.4f}  separation {separation:.4f}")

    # 7. Overlap with our dataset -------------------------------------------
    add(f"\n7. OVERLAP WITH {rel(EXISTING_DATASET)} ({existing.rows:,} rows, unmodified)")
    urls = {r.url for r in valid}
    hosts = {r.hostname for r in valid}
    groups = {r.group for r in valid}
    url_overlap = urls & set(existing.url_labels)
    host_overlap = hosts & set(existing.host_labels)
    group_overlap = groups & set(existing.group_labels)
    add(f"  Normalized URL overlap:          {len(url_overlap):,} of {len(urls):,} distinct CompPhish URLs")
    add(f"  Hostname overlap:                {len(host_overlap):,} of {len(hosts):,} distinct CompPhish hosts")
    add(f"  Registrable-domain overlap:      {len(group_overlap):,} of {len(groups):,} distinct CompPhish domains")
    for label in (1, 0):
        label_groups = {r.group for r in valid if r.label == label}
        overlap = label_groups & set(existing.group_labels)
        same = sum(1 for g in overlap if existing.group_labels[g] == {label})
        add(f"    CompPhish label {label} domains also in ours: {len(overlap):,} "
            f"(ours has only the same label for {same:,})")
    cp_group_labels: dict[str, set[int]] = {}
    for r in valid:
        cp_group_labels.setdefault(r.group, set()).add(r.label)
    disagreeing = sorted(g for g in group_overlap if cp_group_labels[g] != existing.group_labels[g])
    group_counts = Counter(r.group for r in valid)
    add(f"  Shared domains whose label sets differ between the datasets: {len(disagreeing):,}")
    add("  (often shared hosting platforms carrying both phishing and legitimate sites)")
    for group in sorted(disagreeing, key=lambda g: -group_counts[g])[:SAMPLE * 3]:
        add(f"    {group}: CompPhish labels {sorted(cp_group_labels[group])} "
            f"({group_counts[group]:,} rows), our labels {sorted(existing.group_labels[group])}")

    # 8. HTML coverage ------------------------------------------------------
    serial_labels = {r.serial: r.label for r in valid if r.serial != "?"}
    coverage = inspect_html(base / "All_HTML.zip", serial_labels)
    add("\n8. HTML FILES (All_HTML.zip, streamed)")
    add(f"  HTML files: {coverage.members:,} (named <serial>.txt: {coverage.named_by_serial:,}; other names: {len(coverage.other_names)})")
    add(f"  Valid mapping rows with a matching HTML file: {coverage.matched:,} of {len(serial_labels):,}")
    add(f"  Mapping rows without HTML: {len(coverage.mapping_without_html):,} {coverage.mapping_without_html[:SAMPLE]}")
    add(f"  HTML files without a mapping row: {coverage.html_without_mapping:,}")
    for label, name in ((1, "phishing"), (0, "legitimate")):
        sizes = coverage.sizes_by_label[label]
        if sizes:
            add(f"  {name}: {len(sizes):,} files | median {statistics.median(sizes):,.0f} bytes | "
                f"< {TINY_HTML_BYTES} bytes: {coverage.tiny_files_by_label[label]:,} | "
                f"HTML markers in first {HTML_SNIFF_BYTES} bytes: {pct(coverage.html_like_by_label[label], len(sizes))}")
    add("  Correspondence relies on the documented naming rule (file name = serial_number);")
    add("  page content was not compared with the URL, so a redirect or error page")
    add("  saved under a serial cannot be detected here.")

    # 9. Comparison ---------------------------------------------------------
    add("\n9. COMPARISON WITH OUR EXISTING DATASET")
    add(f"  {'':40}{'CompPhish':>16}{'ours':>16}")
    add(f"  {'valid labelled rows':<40}{len(valid):>16,}{existing.rows:>16,}")
    add(f"  {'phishing share':<40}{pct(by_label[1], len(valid)):>16}{pct(existing.labels[1], existing.rows):>16}")
    for label, name in ((1, "phishing"), (0, "legitimate")):
        cp = [r for r in valid if r.label == label]
        ours_n = existing.labels[label]
        add(f"  {name + ' https share':<40}{pct(sum(r.scheme == 'https' for r in cp), len(cp)):>16}"
            f"{pct(existing.https_by_label[(label, True)], ours_n):>16}")
        add(f"  {name + ' homepage share':<40}{pct(sum(r.shape == 'homepage' for r in cp), len(cp)):>16}"
            f"{pct(existing.shape_by_label[(label, 'homepage')], ours_n):>16}")
        add(f"  {name + ' URL-with-path share':<40}{pct(sum(r.shape == 'path' for r in cp), len(cp)):>16}"
            f"{pct(existing.shape_by_label[(label, 'path')], ours_n):>16}")
    add("  Collection period: CompPhish Sep 2024 - Aug 2025 (README); ours: not documented by source.")
    add("  CompPhish includes page HTML; ours is URL-only.")

    # 10. Limitations -------------------------------------------------------
    add("\n10. LIMITATIONS AND UNCERTAINTIES")
    add("  * Label mapping comes from the README and is supported by popularity and")
    add("    keyword statistics; individual labels were not verified.")
    add("  * Per the creator code, entries supplied as bare domains were converted to")
    add("    URLs (https tried first, final URL after redirects kept). The original input")
    add("    list is not included, so which rows were converted is unknown; the high")
    add("    legitimate homepage share suggests most legitimate rows were. For those")
    add("    rows, scheme and path reflect the crawl, not real user browsing.")
    add(f"  * {sum(coverage.tiny_files_by_label.values()):,} HTML files are under {TINY_HTML_BYTES} bytes "
        "(possible error, parking or redirect")
    add("    pages); HTML content quality was not assessed beyond a markup check.")
    add("  * Only sites reachable during crawling were kept (survivorship bias).")
    add("  * HTML is unsanitised third-party content; it was read as bytes only and")
    add("    must not be rendered or opened in a browser.")
    add("  * Registrable domains use the backend's small public-suffix list, so a few")
    add("    domain-overlap counts may be approximate.")
    add("  * The 70 precomputed CompPhish features were not validated or used.")
    add("\n" + rule)

    report = "\n".join(out) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    stdout.write(report)
    stdout.write(f"Report written to {rel(REPORT_PATH)}\n")
    stdout.flush()


if __name__ == "__main__":
    main()
