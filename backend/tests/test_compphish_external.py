"""Tests for preparing a leakage-safe CompPhish external evaluation set."""

import csv
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.features import FEATURE_NAMES, extract_url_features  # noqa: E402
from scripts.prepare_compphish_external_test import (  # noqa: E402
    OUTPUT_FIELDNAMES,
    CompPhishRecord,
    format_report,
    load_compphish_mapping,
    load_existing_domain_groups,
    prepare_external_test,
    write_external_test,
)


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def write_mapping_workbook(path: Path, rows: list[list[str]]) -> None:
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationship_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_ns = "http://schemas.openxmlformats.org/package/2006/relationships"

    workbook = ET.Element(
        "{" + spreadsheet_ns + "}workbook",
        {"xmlns:r": relationship_ns},
    )
    sheets = ET.SubElement(workbook, "{" + spreadsheet_ns + "}sheets")
    ET.SubElement(
        sheets,
        "{" + spreadsheet_ns + "}sheet",
        {"name": "Mapping", "sheetId": "1", "{" + relationship_ns + "}id": "rId1"},
    )

    relationships = ET.Element("{" + package_ns + "}Relationships")
    ET.SubElement(
        relationships,
        "{" + package_ns + "}Relationship",
        {
            "Id": "rId1",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
            "Target": "worksheets/sheet1.xml",
        },
    )

    worksheet = ET.Element("{" + spreadsheet_ns + "}worksheet")
    sheet_data = ET.SubElement(worksheet, "{" + spreadsheet_ns + "}sheetData")
    for row_number, values in enumerate(rows, start=1):
        row_element = ET.SubElement(
            sheet_data,
            "{" + spreadsheet_ns + "}row",
            {"r": str(row_number)},
        )
        for column_number, value in enumerate(values, start=1):
            cell = ET.SubElement(
                row_element,
                "{" + spreadsheet_ns + "}c",
                {"r": f"{_column_name(column_number)}{row_number}", "t": "inlineStr"},
            )
            inline = ET.SubElement(cell, "{" + spreadsheet_ns + "}is")
            text = ET.SubElement(inline, "{" + spreadsheet_ns + "}t")
            text.text = value

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook_zip:
        workbook_zip.writestr(
            "xl/workbook.xml",
            ET.tostring(workbook, encoding="utf-8", xml_declaration=True),
        )
        workbook_zip.writestr(
            "xl/_rels/workbook.xml.rels",
            ET.tostring(relationships, encoding="utf-8", xml_declaration=True),
        )
        workbook_zip.writestr(
            "xl/worksheets/sheet1.xml",
            ET.tostring(worksheet, encoding="utf-8", xml_declaration=True),
        )


def write_existing_dataset(path: Path) -> None:
    path.write_text(
        "url,domain,label\n"
        "https://www.pku.edu.cn/login,stored-wrong-domain.example,0\n"
        "http://192.0.2.1/login,also-wrong.example,1\n",
        encoding="utf-8",
    )


def test_mapping_reader_uses_raw_columns_and_preserves_record_id(tmp_path: Path) -> None:
    mapping = tmp_path / "Mapping_File.xlsx"
    write_mapping_workbook(
        mapping,
        [
            ["serial_number", "url", "label", "url_length"],
            ["101.0", "https://a.example/login", "1.0", "999"],
            ["102", "https://b.example/", "0", "888"],
        ],
    )

    sheet_name, records = load_compphish_mapping(mapping)

    assert sheet_name == "Mapping"
    assert [(record.source_record_id, record.raw_url, record.raw_label) for record in records] == [
        ("101", "https://a.example/login", "1.0"),
        ("102", "https://b.example/", "0"),
    ]


def test_recalculates_both_datasets_and_excludes_overlap_and_duplicates(
    tmp_path: Path,
) -> None:
    existing_csv = tmp_path / "existing.csv"
    write_existing_dataset(existing_csv)
    existing = load_existing_domain_groups(existing_csv)
    records = [
        CompPhishRecord(2, "101", "https://tenant.blogspot.com/path", "1"),
        CompPhishRecord(3, "102", "https://fudan.edu.cn/", "0"),
        CompPhishRecord(4, "103", "https://www.pku.edu.cn/phishing-path", "1"),
        CompPhishRecord(5, "104", "http://192.0.2.1/other", "0"),
        CompPhishRecord(6, "105", "https://Same.example/a", "1"),
        CompPhishRecord(7, "106", "https://same.example/a", "1"),
        CompPhishRecord(8, "107", "https://conflict.example/path", "0"),
        CompPhishRecord(9, "108", "HTTPS://CONFLICT.EXAMPLE/path", "1"),
        CompPhishRecord(10, "109", "https://missing.example", ""),
        CompPhishRecord(11, "110", "https://unsupported.example", "2"),
        CompPhishRecord(12, "111", "  ", "1"),
        CompPhishRecord(13, "112", "javascript:alert(1)", "1"),
    ]

    assert existing.row_count == 2
    assert existing.keys == frozenset({"pku.edu.cn", "192.0.2.1"})

    result = prepare_external_test(records, existing.keys, "2026-11-20T00:00:00Z")

    assert result.input_count == 12
    assert result.valid_record_count == 8
    assert result.overlapping_record_count == 2
    assert result.duplicate_normalized_url_group_count == 2
    assert result.duplicate_normalized_url_record_count == 2
    assert result.conflicting_normalized_url_group_count == 1
    assert result.conflicting_normalized_url_record_count == 2
    assert len(result.rows) == 3
    assert [row["url"] for row in result.rows] == [
        "https://tenant.blogspot.com/path",
        "https://fudan.edu.cn/",
        "https://same.example/a",
    ]
    assert [row["label"] for row in result.rows] == [1, 0, 1]
    assert [row["source_record_id"] for row in result.rows] == ["101", "102", "105"]
    assert set(result.rows[0]) == set(OUTPUT_FIELDNAMES)
    assert tuple(name for name in result.rows[0] if name in FEATURE_NAMES) == FEATURE_NAMES
    assert result.rows[0]["num_digits"] == extract_url_features(
        "https://tenant.blogspot.com/path"
    ).num_digits
    assert result.rows[0]["num_digits"] != 999

    reason_counts = {}
    for exclusion in result.exclusions:
        reason_counts[exclusion.reason] = reason_counts.get(exclusion.reason, 0) + 1
    assert reason_counts == {
        "registrable_domain_overlap": 2,
        "duplicate_normalized_url": 1,
        "conflicting_normalized_url_labels": 2,
        "missing_label": 1,
        "unsupported_label": 1,
        "empty_url": 1,
        "invalid_url": 1,
    }


def test_output_and_report_are_traceable_and_describe_evaluation_limits(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "Mapping_File.xlsx"
    existing_csv = tmp_path / "existing.csv"
    output = tmp_path / "external.csv"
    write_mapping_workbook(
        mapping,
        [
            ["serial_number", "url", "label"],
            ["201", "https://tenant.blogspot.com/a", "1"],
            ["202", "http://fudan.edu.cn/", "0"],
        ],
    )
    write_existing_dataset(existing_csv)
    sheet_name, records = load_compphish_mapping(mapping)
    existing = load_existing_domain_groups(existing_csv)
    result = prepare_external_test(records, existing.keys, "2026-11-20T00:00:00Z")

    assert write_external_test(result.rows, output) == 2
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == OUTPUT_FIELDNAMES
    assert [row["source"] for row in rows] == ["compphish", "compphish"]
    assert [row["source_record_id"] for row in rows] == ["201", "202"]

    report = format_report(
        result,
        existing,
        mapping,
        existing_csv,
        sheet_name,
        "2026-11-20T00:00:00Z",
    )
    assert "External evaluation set; not a verified ground-truth dataset." in report
    assert "Existing dataset column used for grouping: url only; stored domain values ignored" in report
    assert "Unique registrable domains (eTLD+1): 2" in report
    assert "HTTPS: 1 (50.00%)" in report
    assert "HTTP:  1 (50.00%)" in report
    assert "homepage" in report
    assert "path" in report
    assert "Retained external evaluation records: 2" in report


def test_invalid_existing_url_fails_closed(tmp_path: Path) -> None:
    existing_csv = tmp_path / "bad_existing.csv"
    existing_csv.write_text(
        "url,domain,label\nnot a URL,stored.example,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot establish leakage groups safely|invalid URL"):
        load_existing_domain_groups(existing_csv)

