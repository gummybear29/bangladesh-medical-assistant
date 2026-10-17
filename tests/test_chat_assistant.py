"""Tests for ConversationalMedicalAssistant (Memory, Antecedent, Strict Grounding, Safety)."""

from __future__ import annotations

import pytest

from src.answer.chat_assistant import ConversationalMedicalAssistant
from src.answer.engine import AnswerEngine
from src.llm.client import FakeLLMClient


@pytest.fixture
def mock_assistant() -> ConversationalMedicalAssistant:
    fake_client = FakeLLMClient(
        default_response="Metformin dosage starts at 500 mg orally once daily. [metformin | dosage_and_administration]"
    )
    engine = AnswerEngine(llm_client=fake_client)
    return ConversationalMedicalAssistant(answer_engine=engine)


def test_safety_escalation_emergency(mock_assistant: ConversationalMedicalAssistant) -> None:
    alerts = mock_assistant.detect_safety_escalations("Patient is bleeding heavily and unresponsive")
    assert len(alerts) > 0
    assert any(a.alert_type == "EMERGENCY" for a in alerts)
    assert "999" in alerts[0].hotlines


def test_safety_escalation_pregnancy(mock_assistant: ConversationalMedicalAssistant) -> None:
    alerts = mock_assistant.detect_safety_escalations("Can a pregnant woman take this medicine?")
    assert len(alerts) > 0
    assert any(a.alert_type == "PREGNANCY" for a in alerts)


def test_safety_escalation_pediatric(mock_assistant: ConversationalMedicalAssistant) -> None:
    alerts = mock_assistant.detect_safety_escalations("What is the dose for a 3 year old child?")
    assert len(alerts) > 0
    assert any(a.alert_type == "PEDIATRIC" for a in alerts)


def test_antecedent_resolution_pronoun(mock_assistant: ConversationalMedicalAssistant) -> None:
    history = [{"user": "What is Napa?", "reply": "Napa is paracetamol.", "active_drug": "paracetamol"}]
    resolved, drug = mock_assistant.resolve_conversation_context("can I take this with paracetamol?", history)
    assert drug == "paracetamol"
    assert "paracetamol" in resolved.lower()


def test_bangladesh_alternatives_intent(mock_assistant: ConversationalMedicalAssistant) -> None:
    history = [{"user": "What is Napa?", "reply": "Napa is paracetamol.", "active_drug": "paracetamol"}]
    res = mock_assistant.respond("What are generic alternatives in BD?", conversation_history=history)
    assert res.query_type == "alternatives"
    assert "Bangladeshi Brand Alternatives" in res.reply_text
    assert "Napa" in res.reply_text
    assert "Ace" in res.reply_text or "Fast" in res.reply_text


def test_brand_identification_intent(mock_assistant: ConversationalMedicalAssistant) -> None:
    res = mock_assistant.respond("What is Napa?")
    assert res.query_type == "identify"
    assert "Paracetamol" in res.reply_text
    assert res.active_drug == "paracetamol"


def test_bengali_brand_identification_seclo(mock_assistant: ConversationalMedicalAssistant) -> None:
    res = mock_assistant.respond("Seclo কী ওষুধ? এর জেনেরিক নাম কী?")
    assert res.query_type == "identify"
    assert res.confidence == "high"
    assert "ওমিপ্রাজল" in res.reply_text
    assert "Omeprazole" in res.reply_text
    assert res.active_drug == "omeprazole"


def test_first_aid_intent(mock_assistant: ConversationalMedicalAssistant) -> None:
    res = mock_assistant.respond("What is the first aid for snakebite?")
    assert res.query_type == "first_aid"
    assert "First-Aid Protocol" in res.reply_text
    assert "999" in res.reply_text


def test_strict_grounding_abstention(mock_assistant: ConversationalMedicalAssistant) -> None:
    res = mock_assistant.respond("What is the dosage of fictitiousexplodium for dragon fever?")
    assert res.confidence == "abstained"
    assert "Information Not Found in Official Drug Labeling" in res.reply_text
