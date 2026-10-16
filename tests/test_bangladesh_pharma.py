"""Tests for Bangladesh Pharmaceutical Directory and Emergency Hotlines."""

from __future__ import annotations

import pytest

from src.knowledge.bangladesh_pharma import (
    BANGLADESH_BRANDS,
    BANGLADESH_EMERGENCY_HOTLINES,
    get_canonical_generic_from_local_query,
    is_emergency_query,
    lookup_bangladesh_brand,
)
from src.retrieval.hybrid import detect_drugs_in_query


def test_emergency_hotlines_defined():
    assert "999" in BANGLADESH_EMERGENCY_HOTLINES
    assert "16263" in BANGLADESH_EMERGENCY_HOTLINES
    assert "01700-700700" in BANGLADESH_EMERGENCY_HOTLINES


def test_lookup_bangladesh_brand():
    napa = lookup_bangladesh_brand("Can I take Napa for headache?")
    assert napa is not None
    assert napa.generic_name == "paracetamol"
    assert "Beximco" in napa.manufacturer

    seclo = lookup_bangladesh_brand("How to take Seclo 20 mg?")
    assert seclo is not None
    assert seclo.generic_name == "omeprazole"
    assert "Square" in seclo.manufacturer


def test_detect_paracitemol_typo_as_acetaminophen():
    canonical = get_canonical_generic_from_local_query("what is paracitemol?")
    assert canonical == "acetaminophen"

    detected = detect_drugs_in_query("what is paracitemol?")
    assert "acetaminophen" in detected


def test_detect_napa_extra_and_ace():
    detected_napa = detect_drugs_in_query("napa extra dose")
    assert "acetaminophen" in detected_napa

    detected_ace = detect_drugs_in_query("ace plus for migraine")
    assert "acetaminophen" in detected_ace


def test_emergency_query_detection():
    is_em, reason = is_emergency_query("Patient has a venomous snakebite on leg")
    assert is_em is True
    assert reason == "snakebite"

    is_em_bn, reason_bn = is_emergency_query("বুকে তীব্র ব্যথা এবং দম বন্ধ হয়ে আসছে")
    assert is_em_bn is True
    assert reason_bn == "cardiac"

    is_em_normal, _ = is_emergency_query("What is the dose of paracetamol?")
    assert is_em_normal is False
