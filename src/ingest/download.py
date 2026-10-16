"""Task 1 — Fetch ~60 common drug labels from openFDA.

Each label is split into five standard sections and saved as a JSON file
under data/labels/<drug_name>.json.

Behaviour:
- Prefers single-ingredient labels over combination products.
- Falls back to the first result (with a log warning) if no single-ingredient
  label is found.
- Tries multiple openFDA fields for warnings:
  warnings → warnings_and_precautions → warnings_and_cautions → boxed_warning.
- Stores set_id and effective_time; builds source_url as a DailyMed link.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

from src.config import (
    LABELS_DIR,
    OPENFDA_BASE_URL,
    OPENFDA_REQUEST_DELAY,
)

logger = logging.getLogger(__name__)

# ── Section mapping ────────────────────────────────────────────────────
# openFDA field name → short key used in our JSON.
# Warnings are handled separately via WARNINGS_FALLBACK.
SECTION_MAP: dict[str, str] = {
    "indications_and_usage": "indications",
    "dosage_and_administration": "dosage",
    "drug_interactions": "interactions",
    "contraindications": "contraindications",
}

# Tried in order; first match wins.
WARNINGS_FALLBACK: list[str] = [
    "warnings",
    "warnings_and_precautions",
    "warnings_and_cautions",
    "boxed_warning",
]

INTERACTIONS_FALLBACK: list[str] = [
    "drug_interactions",
    "ask_doctor_or_pharmacist",
]

CONTRAINDICATIONS_FALLBACK: list[str] = [
    "contraindications",
    "do_not_use",
    "stop_use",
]

# ── Drug list (~60 common generics) ───────────────────────────────────
DRUG_LIST: list[str] = [
    "acetaminophen",
    "albuterol",
    "alendronate",
    "allopurinol",
    "alprazolam",
    "amitriptyline",
    "amlodipine",
    "amoxicillin",
    "atorvastatin",
    "azithromycin",
    "carvedilol",
    "cephalexin",
    "ciprofloxacin",
    "citalopram",
    "clonazepam",
    "clopidogrel",
    "cyclobenzaprine",
    "diazepam",
    "diclofenac",
    "diltiazem",
    "doxycycline",
    "duloxetine",
    "escitalopram",
    "fluoxetine",
    "furosemide",
    "gabapentin",
    "hydrochlorothiazide",
    "hydroxychloroquine",
    "ibuprofen",
    "insulin glargine",
    "lamotrigine",
    "levothyroxine",
    "lisinopril",
    "loratadine",
    "lorazepam",
    "losartan",
    "meloxicam",
    "metformin",
    "metoprolol",
    "metronidazole",
    "montelukast",
    "naproxen",
    "omeprazole",
    "ondansetron",
    "oxycodone",
    "pantoprazole",
    "paroxetine",
    "prednisone",
    "pregabalin",
    "propranolol",
    "rosuvastatin",
    "sertraline",
    "simvastatin",
    "spironolactone",
    "tamsulosin",
    "tramadol",
    "trazodone",
    "valsartan",
    "venlafaxine",
    "warfarin",
    "zolpidem",
]

assert len(DRUG_LIST) >= 60, f"Need ≥60 drugs, got {len(DRUG_LIST)}"


# ── Helpers ────────────────────────────────────────────────────────────


def _build_api_url(drug_name: str) -> str:
    """Return the openFDA query URL for a drug's label (fallback URL)."""
    return (
        f'{OPENFDA_BASE_URL}?search=openfda.generic_name:"{drug_name}"&limit=1'
    )


def _dailymed_url(set_id: str) -> str:
    """Return a DailyMed deep-link for a given SPL set_id."""
    return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"


def _is_single_ingredient(result: dict) -> bool:
    """Return True if the label appears to be for a single active ingredient.

    Checks ``openfda.substance_name`` first (most reliable); falls back to
    ``openfda.generic_name`` heuristics.
    """
    openfda = result.get("openfda", {})

    # Prefer substance_name — each entry is one ingredient
    substances = openfda.get("substance_name", [])
    if substances:
        return len(substances) == 1

    # Fallback: generic_name list with one entry that has no comma / " AND "
    generic_names = openfda.get("generic_name", [])
    if generic_names and len(generic_names) == 1:
        name = generic_names[0].upper()
        return "," not in name and " AND " not in name

    return False


def _is_prescription(result: dict) -> bool:
    """Return True if the label is for a human prescription drug."""
    openfda = result.get("openfda", {})
    product_types = openfda.get("product_type", [])
    return any("HUMAN PRESCRIPTION DRUG" in pt.upper() for pt in product_types)


def _extract_first(raw_label: dict, fields: list[str]) -> str | None:
    """Extract first non-empty text from a list of candidate fields."""
    for field in fields:
        value = raw_label.get(field)
        if isinstance(value, list) and value:
            return value[0]
        elif isinstance(value, str) and value.strip():
            return value
    return None


# ── Core functions ─────────────────────────────────────────────────────


def fetch_label(drug_name: str) -> dict | None:
    """Fetch a single drug label from openFDA.

    Retrieves up to 10 results and prioritizes:
    1. Single-ingredient prescription drug (HUMAN PRESCRIPTION DRUG)
    2. Single-ingredient drug (e.g. OTC)
    3. Fallback to first result (with warning log if combination product).

    Returns ``None`` on any hard failure (HTTP error, network error, no
    results at all).
    """
    url = OPENFDA_BASE_URL
    params = {
        "search": f'openfda.generic_name:"{drug_name}"',
        "limit": "10",
    }
    try:
        resp = httpx.get(url, params=params, timeout=15.0)
        if resp.status_code != 200:
            logger.warning(
                "openFDA HTTP %d for %s", resp.status_code, drug_name,
            )
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            logger.warning("No results for %s", drug_name)
            return None

        # 1. Prefer single-ingredient prescription label
        for result in results:
            if _is_single_ingredient(result) and _is_prescription(result):
                return result

        # 2. Prefer any single-ingredient label
        for result in results:
            if _is_single_ingredient(result):
                return result

        # 3. Fallback — log combo usage
        logger.warning(
            "No single-ingredient label for %s; using first result "
            "(combination product)",
            drug_name,
        )
        return results[0]

    except (httpx.HTTPError, Exception) as exc:
        logger.error("Failed to fetch %s: %s", drug_name, exc)
        return None


def parse_sections(raw_label: dict) -> dict[str, str | None]:
    """Extract the five target sections from a raw openFDA label result.

    Applies fallback chains for warnings, interactions (including OTC
    ask_doctor_or_pharmacist), and contraindications (including OTC
    do_not_use and stop_use). Missing sections become ``None``.
    """
    sections: dict[str, str | None] = {}

    # Indications & dosage
    for fda_key, short_key in [
        ("indications_and_usage", "indications"),
        ("dosage_and_administration", "dosage"),
    ]:
        val = raw_label.get(fda_key)
        sections[short_key] = val[0] if isinstance(val, list) and val else None

    # Warnings fallback chain
    sections["warnings"] = _extract_first(raw_label, WARNINGS_FALLBACK)

    # Interactions fallback chain (including OTC ask_doctor_or_pharmacist)
    sections["interactions"] = _extract_first(raw_label, INTERACTIONS_FALLBACK)

    # Contraindications fallback chain (including OTC do_not_use / stop_use)
    sections["contraindications"] = _extract_first(
        raw_label, CONTRAINDICATIONS_FALLBACK
    )

    return sections


def download_all() -> list[str]:
    """Download labels for all drugs in DRUG_LIST.

    Returns a list of drug names that were successfully saved.
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for i, drug in enumerate(DRUG_LIST):
        logger.info("[%d/%d] Fetching %s …", i + 1, len(DRUG_LIST), drug)
        raw = fetch_label(drug)
        if raw is None:
            logger.warning("Skipping %s (no data)", drug)
            continue

        sections = parse_sections(raw)
        set_id = raw.get("set_id")
        effective_time = raw.get("effective_time")

        source_url = _dailymed_url(set_id) if set_id else _build_api_url(drug)

        record = {
            "drug_name": drug,
            "set_id": set_id,
            "effective_time": effective_time,
            "sections": sections,
            "source_url": source_url,
        }

        out_path = LABELS_DIR / f"{drug}.json"
        out_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved.append(drug)
        logger.info("Saved %s", out_path.name)

        # Rate-limit to respect openFDA (skipped in tests via patch)
        if i < len(DRUG_LIST) - 1:
            time.sleep(OPENFDA_REQUEST_DELAY)

    logger.info("Done: %d / %d drugs saved.", len(saved), len(DRUG_LIST))
    return saved


# ── CLI entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    download_all()
