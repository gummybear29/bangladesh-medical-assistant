"""Tests for Multimodal Medical Assistant (Vision + First-Aid + BD Pharma)."""

from __future__ import annotations

import io
import pytest
from PIL import Image

from src.llm.multimodal import MultimodalMedicalAssistant


def test_multimodal_text_only_bd_brand():
    assistant = MultimodalMedicalAssistant(api_key="")  # offline mode
    res = assistant.analyze(prompt="What is Napa 500 used for?", language="en")
    assert "Napa" in res.text
    assert "Paracetamol" in res.text or "Beximco" in res.text
    assert "16263" in res.text


def test_multimodal_first_aid_protocol():
    assistant = MultimodalMedicalAssistant(api_key="")
    res = assistant.analyze(prompt="What to do for snakebite emergency?", language="en")
    assert "Snakebite" in res.text
    assert "999" in res.text
    assert res.emergency_level == "EMERGENCY"


def test_multimodal_image_analysis_offline():
    assistant = MultimodalMedicalAssistant(api_key="")
    # Create simple dummy image
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))
    res = assistant.analyze(prompt="Inspect this cut", image=img, language="en")
    assert "First Aid Protocol" in res.text or "Bleeding" in res.text
    assert "Uploaded Medical Image" in res.identified_items


def test_multimodal_bengali_language():
    assistant = MultimodalMedicalAssistant(api_key="")
    res = assistant.analyze(prompt="আগুনে পোড়ার প্রাথমিক চিকিৎসা কী?", language="bn")
    assert "প্রাথমিক চিকিৎসা" in res.text
    assert "পানি" in res.text
