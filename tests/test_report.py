"""Tests for src.ingest.report.

Tests label completeness reporting, section counting, and flagging logic.
"""

import json
from pathlib import Path

from src.ingest.report import generate_report


def test_empty_directory(tmp_path: Path, capsys):
    summary = generate_report(tmp_path)
    captured = capsys.readouterr()
    assert summary["total"] == 0
    assert summary["flagged"] == 0
    assert "No label files found" in captured.out


def test_generate_report_flags_under_four_sections(tmp_path: Path, capsys):
    # Drug 1: 5/5 sections
    d1 = {
        "drug_name": "drug_full",
        "set_id": "id-1",
        "sections": {
            "indications": "ind",
            "dosage": "dos",
            "warnings": "wrn",
            "interactions": "int",
            "contraindications": "ctr",
        },
    }
    # Drug 2: 3/5 sections (missing interactions, contraindications)
    d2 = {
        "drug_name": "drug_partial",
        "set_id": "id-2",
        "sections": {
            "indications": "ind",
            "dosage": "dos",
            "warnings": "wrn",
            "interactions": None,
            "contraindications": None,
        },
    }
    # Drug 3: 4/5 sections (missing contraindications)
    d3 = {
        "drug_name": "drug_four",
        "set_id": "id-3",
        "sections": {
            "indications": "ind",
            "dosage": "dos",
            "warnings": "wrn",
            "interactions": "int",
            "contraindications": None,
        },
    }

    (tmp_path / "drug_full.json").write_text(json.dumps(d1), encoding="utf-8")
    (tmp_path / "drug_partial.json").write_text(json.dumps(d2), encoding="utf-8")
    (tmp_path / "drug_four.json").write_text(json.dumps(d3), encoding="utf-8")

    summary = generate_report(tmp_path)
    captured = capsys.readouterr()

    assert summary["total"] == 3
    assert summary["total_missing"] == 3  # 0 + 2 + 1
    assert summary["flagged"] == 1
    assert summary["flagged_drugs"] == ["drug_partial"]

    assert "drug_full" in captured.out
    assert "drug_partial" in captured.out
    assert "<<" in captured.out
    assert ">> drug_partial: missing interactions, contraindications" in captured.out
