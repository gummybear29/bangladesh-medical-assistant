"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_labels_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for label JSON files."""
    labels = tmp_path / "labels"
    labels.mkdir()
    return labels


def make_openfda_response(
    drug_name: str,
    *,
    indications: str | None = "Treat condition X.",
    dosage: str | None = "Take 500 mg daily.",
    warnings: str | None = "May cause dizziness.",
    interactions: str | None = "Avoid with drug Y.",
    contraindications: str | None = "Do not use if allergic.",
    set_id: str = "test-set-id-1234",
    effective_time: str = "20240101",
    substance_names: list[str] | None = None,
) -> dict:
    """Build a realistic openFDA API response for a single drug label.

    By default the response represents a single-ingredient product
    (one substance_name entry).
    """
    result: dict = {}
    if indications is not None:
        result["indications_and_usage"] = [indications]
    if dosage is not None:
        result["dosage_and_administration"] = [dosage]
    if warnings is not None:
        result["warnings"] = [warnings]
    if interactions is not None:
        result["drug_interactions"] = [interactions]
    if contraindications is not None:
        result["contraindications"] = [contraindications]

    result["set_id"] = set_id
    result["effective_time"] = effective_time

    openfda: dict = {"generic_name": [drug_name.upper()]}
    if substance_names is not None:
        openfda["substance_name"] = substance_names
    else:
        openfda["substance_name"] = [drug_name.upper()]
    result["openfda"] = openfda

    return {"results": [result]}
