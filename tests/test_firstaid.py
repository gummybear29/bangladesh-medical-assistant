"""Tests for Bangladesh First-Aid and Emergency Triage Protocols."""

from __future__ import annotations

import pytest

from src.firstaid.triage import FIRST_AID_PROTOCOLS, find_first_aid_protocol


def test_core_protocols_present():
    expected = ["dengue", "burns", "snakebite", "diarrhea", "bleeding", "choking", "fracture"]
    for p in expected:
        assert p in FIRST_AID_PROTOCOLS


def test_dengue_protocol_forbids_nsaids():
    dengue = FIRST_AID_PROTOCOLS["dengue"]
    assert any("Paracetamol" in do for do in dengue.dos_en)
    assert any("Aspirin" in dont or "Ibuprofen" in dont for dont in dengue.donts_en)
    assert any("999" in c or "16263" in c for c in dengue.emergency_contacts)


def test_burns_protocol_forbids_toothpaste_and_ice():
    burns = FIRST_AID_PROTOCOLS["burns"]
    assert any("running" in do.lower() for do in burns.dos_en)
    assert any("toothpaste" in dont.lower() for dont in burns.donts_en)
    assert any("ice" in dont.lower() for dont in burns.donts_en)


def test_snakebite_protocol_forbids_cutting_and_tourniquets():
    snake = FIRST_AID_PROTOCOLS["snakebite"]
    assert snake.urgency_level == "EMERGENCY_CRITICAL"
    assert any("cut" in dont.lower() for dont in snake.donts_en)
    assert any("tourniquet" in dont.lower() for dont in snake.donts_en)
    assert any("splint" in do.lower() or "still" in do.lower() for do in snake.dos_en)


def test_find_first_aid_protocol():
    res_dengue = find_first_aid_protocol("Patient has dengue fever and body pain")
    assert res_dengue is not None
    assert res_dengue.id == "dengue"

    res_burn = find_first_aid_protocol("Child got hot water burn on arm")
    assert res_burn is not None
    assert res_burn.id == "burns"

    res_snake = find_first_aid_protocol("সাপের কামড় দিলে কী করব?")
    assert res_snake is not None
    assert res_snake.id == "snakebite"

    res_bone = find_first_aid_protocol("how to fix bone")
    assert res_bone is not None
    assert res_bone.id == "fracture"
    assert any("still" in do.lower() for do in res_bone.dos_en)
    assert any("straighten" in dont.lower() or "push" in dont.lower() for dont in res_bone.donts_en)
    assert any("NITOR" in c for c in res_bone.emergency_contacts)
