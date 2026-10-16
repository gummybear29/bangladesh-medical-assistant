"""Tests for src.ingest.download — Task 1.

All openFDA calls are mocked via respx. No live network traffic.
"""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from tests.conftest import make_openfda_response
from src.ingest.download import (
    DRUG_LIST,
    SECTION_MAP,
    WARNINGS_FALLBACK,
    INTERACTIONS_FALLBACK,
    CONTRAINDICATIONS_FALLBACK,
    fetch_label,
    parse_sections,
    download_all,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _mock_route(drug_name: str, response_body: dict, status: int = 200):
    """Register a respx route for a given drug name."""
    respx.get(
        "https://api.fda.gov/drug/label.json",
        params__contains={"search": f'openfda.generic_name:"{drug_name}"'},
    ).mock(return_value=httpx.Response(status, json=response_body))


def _make_combo_result(drug_name: str, *, set_id: str = "combo-id") -> dict:
    """Build a raw openFDA result dict for a combination product."""
    return {
        "set_id": set_id,
        "effective_time": "20240101",
        "indications_and_usage": ["Combo indication text."],
        "dosage_and_administration": ["Combo dosage text."],
        "warnings": ["Combo warnings."],
        "drug_interactions": ["Combo interactions."],
        "contraindications": ["Combo contraindications."],
        "openfda": {
            "generic_name": [
                f"{drug_name.upper()} HYDROCHLORIDE, SITAGLIPTIN PHOSPHATE"
            ],
            "substance_name": [
                f"{drug_name.upper()} HYDROCHLORIDE",
                "SITAGLIPTIN PHOSPHATE",
            ],
            "product_type": ["HUMAN PRESCRIPTION DRUG"],
        },
    }


def _make_single_result(
    drug_name: str,
    *,
    set_id: str = "single-id",
    product_type: str = "HUMAN PRESCRIPTION DRUG",
) -> dict:
    """Build a raw openFDA result dict for a single-ingredient product."""
    return {
        "set_id": set_id,
        "effective_time": "20240201",
        "indications_and_usage": ["Single-ingredient indication."],
        "dosage_and_administration": ["Single dosage."],
        "warnings": ["Single warnings."],
        "drug_interactions": ["Single interactions."],
        "contraindications": ["Single contraindications."],
        "openfda": {
            "generic_name": [f"{drug_name.upper()} HYDROCHLORIDE"],
            "substance_name": [f"{drug_name.upper()} HYDROCHLORIDE"],
            "product_type": [product_type],
        },
    }


# ── Drug list sanity ──────────────────────────────────────────────────


class TestDrugList:
    def test_has_at_least_60_drugs(self):
        assert len(DRUG_LIST) >= 60

    def test_all_lowercase(self):
        for drug in DRUG_LIST:
            assert drug == drug.lower(), f"{drug} must be lowercase"

    def test_no_duplicates(self):
        assert len(DRUG_LIST) == len(set(DRUG_LIST))


# ── Section map ───────────────────────────────────────────────────────


class TestSectionMap:
    def test_has_four_non_warnings_sections(self):
        assert set(SECTION_MAP.values()) == {
            "indications",
            "dosage",
            "interactions",
            "contraindications",
        }

    def test_warnings_fallback_has_correct_order(self):
        assert WARNINGS_FALLBACK == [
            "warnings",
            "warnings_and_precautions",
            "warnings_and_cautions",
            "boxed_warning",
        ]

    def test_interactions_fallback_order(self):
        assert INTERACTIONS_FALLBACK == [
            "drug_interactions",
            "ask_doctor_or_pharmacist",
        ]

    def test_contraindications_fallback_order(self):
        assert CONTRAINDICATIONS_FALLBACK == [
            "contraindications",
            "do_not_use",
            "stop_use",
        ]


# ── fetch_label — basics ─────────────────────────────────────────────


class TestFetchLabel:
    @respx.mock
    def test_returns_result_on_success(self):
        body = make_openfda_response("metformin")
        _mock_route("metformin", body)
        result = fetch_label("metformin")
        assert result is not None
        assert "indications_and_usage" in result

    @respx.mock
    def test_returns_none_on_404(self):
        _mock_route("fakedrug", {"error": {"code": "NOT_FOUND"}}, status=404)
        assert fetch_label("fakedrug") is None

    @respx.mock
    def test_returns_none_on_empty_results(self):
        _mock_route("emptydrug", {"results": []})
        assert fetch_label("emptydrug") is None

    @respx.mock
    def test_returns_none_on_network_error(self):
        respx.get(
            "https://api.fda.gov/drug/label.json",
            params__contains={"search": 'openfda.generic_name:"baddrug"'},
        ).mock(side_effect=httpx.ConnectError("connection refused"))
        assert fetch_label("baddrug") is None


# ── fetch_label — single-ingredient preference ───────────────────────


class TestSingleIngredientPreference:
    @respx.mock
    def test_prefers_single_over_combo(self):
        """When both combo and single labels exist, pick the single one."""
        combo = _make_combo_result("metformin", set_id="combo-set-id")
        single = _make_single_result("metformin", set_id="single-set-id")
        _mock_route("metformin", {"results": [combo, single]})

        result = fetch_label("metformin")
        assert result is not None
        assert result["set_id"] == "single-set-id"

    @respx.mock
    def test_falls_back_to_first_when_only_combos(self):
        """If no single-ingredient label exists, use first result and log."""
        combo = _make_combo_result("metformin", set_id="only-combo")
        _mock_route("metformin", {"results": [combo]})

        result = fetch_label("metformin")
        assert result is not None
        assert result["set_id"] == "only-combo"

    @respx.mock
    def test_detects_single_via_substance_name(self):
        """Single substance_name entry → single-ingredient."""
        result_data = _make_single_result("aspirin")
        _mock_route("aspirin", {"results": [result_data]})

        result = fetch_label("aspirin")
        assert result is not None

    @respx.mock
    def test_detects_combo_via_multiple_substance_names(self):
        """Multiple substance_name entries → combination product."""
        combo = _make_combo_result("losartan", set_id="combo")
        single = _make_single_result("losartan", set_id="single")
        _mock_route("losartan", {"results": [combo, single]})

        result = fetch_label("losartan")
        assert result["set_id"] == "single"

    @respx.mock
    def test_prefers_prescription_over_otc_single_ingredient(self):
        """When single-ingredient Rx and OTC labels exist, prefer the Rx label."""
        otc = _make_single_result(
            "omeprazole", set_id="otc-id", product_type="HUMAN OTC DRUG"
        )
        rx = _make_single_result(
            "omeprazole", set_id="rx-id", product_type="HUMAN PRESCRIPTION DRUG"
        )
        _mock_route("omeprazole", {"results": [otc, rx]})

        result = fetch_label("omeprazole")
        assert result is not None
        assert result["set_id"] == "rx-id"

    @respx.mock
    def test_falls_back_to_otc_when_no_prescription_single_ingredient(self):
        """If only OTC single-ingredient exists, select the OTC label."""
        otc = _make_single_result(
            "acetaminophen", set_id="otc-id", product_type="HUMAN OTC DRUG"
        )
        _mock_route("acetaminophen", {"results": [otc]})

        result = fetch_label("acetaminophen")
        assert result is not None
        assert result["set_id"] == "otc-id"


# ── parse_sections — standard fields ─────────────────────────────────


class TestParseSections:
    def test_extracts_all_five_sections(self):
        body = make_openfda_response("ibuprofen")
        raw = body["results"][0]
        sections = parse_sections(raw)
        assert set(sections.keys()) == {
            "indications", "dosage", "warnings", "interactions", "contraindications",
        }
        assert sections["indications"] == "Treat condition X."

    def test_missing_section_becomes_none(self):
        body = make_openfda_response("partial", interactions=None, warnings=None)
        raw = body["results"][0]
        sections = parse_sections(raw)
        assert sections["interactions"] is None
        assert sections["warnings"] is None
        assert sections["indications"] == "Treat condition X."

    def test_all_sections_missing(self):
        raw = {"openfda": {"generic_name": ["emptydrug"]}}
        sections = parse_sections(raw)
        for v in sections.values():
            assert v is None


# ── parse_sections — warnings fallback chain ─────────────────────────


class TestWarningsFallback:
    def test_prefers_warnings_field(self):
        raw = {
            "warnings": ["Direct warnings text."],
            "warnings_and_precautions": ["W&P text."],
        }
        sections = parse_sections(raw)
        assert sections["warnings"] == "Direct warnings text."

    def test_falls_back_to_warnings_and_precautions(self):
        raw = {"warnings_and_precautions": ["W&P fallback."]}
        sections = parse_sections(raw)
        assert sections["warnings"] == "W&P fallback."

    def test_falls_back_to_warnings_and_cautions(self):
        raw = {"warnings_and_cautions": ["W&C fallback."]}
        sections = parse_sections(raw)
        assert sections["warnings"] == "W&C fallback."

    def test_falls_back_to_boxed_warning(self):
        raw = {"boxed_warning": ["Boxed warning text."]}
        sections = parse_sections(raw)
        assert sections["warnings"] == "Boxed warning text."

    def test_none_when_no_warnings_field_present(self):
        raw = {"indications_and_usage": ["Something."]}
        sections = parse_sections(raw)
        assert sections["warnings"] is None

    def test_respects_priority_order(self):
        """boxed_warning alone should NOT win over warnings_and_cautions."""
        raw = {
            "boxed_warning": ["Boxed."],
            "warnings_and_cautions": ["W&C."],
        }
        sections = parse_sections(raw)
        assert sections["warnings"] == "W&C."


# ── parse_sections — OTC field mapping ───────────────────────────────


class TestOtcFieldMapping:
    def test_interactions_prefers_drug_interactions(self):
        raw = {
            "drug_interactions": ["Rx interactions text."],
            "ask_doctor_or_pharmacist": ["OTC ask doctor text."],
        }
        sections = parse_sections(raw)
        assert sections["interactions"] == "Rx interactions text."

    def test_interactions_falls_back_to_ask_doctor_or_pharmacist(self):
        raw = {
            "ask_doctor_or_pharmacist": ["Ask doctor if taking blood thinners."],
        }
        sections = parse_sections(raw)
        assert sections["interactions"] == "Ask doctor if taking blood thinners."

    def test_contraindications_prefers_contraindications(self):
        raw = {
            "contraindications": ["Do not administer to patients with..."],
            "do_not_use": ["Do not use if allergic."],
            "stop_use": ["Stop use if rash appears."],
        }
        sections = parse_sections(raw)
        assert (
            sections["contraindications"]
            == "Do not administer to patients with..."
        )

    def test_contraindications_falls_back_to_do_not_use(self):
        raw = {
            "do_not_use": ["Do not use if you have kidney disease."],
            "stop_use": ["Stop use if pain worsens."],
        }
        sections = parse_sections(raw)
        assert (
            sections["contraindications"]
            == "Do not use if you have kidney disease."
        )

    def test_contraindications_falls_back_to_stop_use(self):
        raw = {
            "stop_use": ["Stop use and ask doctor if fever lasts >3 days."],
        }
        sections = parse_sections(raw)
        assert (
            sections["contraindications"]
            == "Stop use and ask doctor if fever lasts >3 days."
        )

    def test_full_otc_label_mapping(self):
        raw = {
            "indications_and_usage": ["Relieves minor aches and pains."],
            "dosage_and_administration": ["Take 2 caplets every 6 hours."],
            "warnings": ["Liver warning: severe liver damage may occur."],
            "ask_doctor_or_pharmacist": ["Ask doctor or pharmacist before use."],
            "do_not_use": [
                "Do not use with any other drug containing acetaminophen."
            ],
        }
        sections = parse_sections(raw)
        assert sections["indications"] == "Relieves minor aches and pains."
        assert sections["dosage"] == "Take 2 caplets every 6 hours."
        assert (
            sections["warnings"]
            == "Liver warning: severe liver damage may occur."
        )
        assert (
            sections["interactions"] == "Ask doctor or pharmacist before use."
        )
        assert (
            sections["contraindications"]
            == "Do not use with any other drug containing acetaminophen."
        )
        assert all(v is not None for v in sections.values())


# ── download_all — integration (fully mocked) ────────────────────────


class TestDownloadAll:
    @respx.mock
    def test_saves_json_with_correct_schema(self, tmp_labels_dir: Path):
        """download_all writes JSON with drug_name, set_id, effective_time,
        sections, and a DailyMed source_url."""
        drugs = ["aspirin", "ibuprofen"]
        for drug in drugs:
            _mock_route(drug, make_openfda_response(drug, set_id=f"{drug}-set-id"))

        with patch("src.ingest.download.DRUG_LIST", drugs), \
             patch("src.ingest.download.LABELS_DIR", tmp_labels_dir), \
             patch("src.ingest.download.OPENFDA_REQUEST_DELAY", 0):
            results = download_all()

        assert len(results) == 2
        for drug in drugs:
            fpath = tmp_labels_dir / f"{drug}.json"
            assert fpath.exists()

            data = json.loads(fpath.read_text(encoding="utf-8"))
            assert data["drug_name"] == drug
            assert data["set_id"] == f"{drug}-set-id"
            assert data["effective_time"] == "20240101"
            assert set(data["sections"].keys()) == {
                "indications", "dosage", "warnings", "interactions",
                "contraindications",
            }
            assert "dailymed.nlm.nih.gov" in data["source_url"]
            assert f"{drug}-set-id" in data["source_url"]

    @respx.mock
    def test_skips_failed_drugs(self, tmp_labels_dir: Path):
        _mock_route("gooddrug", make_openfda_response("gooddrug"))
        _mock_route("baddrug", {"error": {"code": "NOT_FOUND"}}, status=404)

        with patch("src.ingest.download.DRUG_LIST", ["gooddrug", "baddrug"]), \
             patch("src.ingest.download.LABELS_DIR", tmp_labels_dir), \
             patch("src.ingest.download.OPENFDA_REQUEST_DELAY", 0):
            results = download_all()

        assert len(results) == 1
        assert (tmp_labels_dir / "gooddrug.json").exists()
        assert not (tmp_labels_dir / "baddrug.json").exists()

    @respx.mock
    def test_source_url_falls_back_without_set_id(self, tmp_labels_dir: Path):
        """If set_id is missing, source_url falls back to the API query URL."""
        body = make_openfda_response("nodrug")
        body["results"][0].pop("set_id")
        _mock_route("nodrug", body)

        with patch("src.ingest.download.DRUG_LIST", ["nodrug"]), \
             patch("src.ingest.download.LABELS_DIR", tmp_labels_dir), \
             patch("src.ingest.download.OPENFDA_REQUEST_DELAY", 0):
            download_all()

        data = json.loads(
            (tmp_labels_dir / "nodrug.json").read_text(encoding="utf-8")
        )
        assert data["set_id"] is None
        assert "api.fda.gov" in data["source_url"]
