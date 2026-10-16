"""Tests for src.llm.client — Task 4.

All tests are completely mocked. No live API calls to Google Gemini.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import (
    FakeLLMClient,
    GeminiClient,
    LLMClient,
    LLMResponse,
)


class TestFakeLLMClient:
    def test_default_canned_response(self):
        client = FakeLLMClient(default_response="Mock answer [metformin | dosage]")
        resp = client.generate("What is the dose?")
        assert resp.success is True
        assert resp.text == "Mock answer [metformin | dosage]"
        assert resp.from_cache is False

    def test_mapped_responses(self):
        client = FakeLLMClient(
            responses={"dose": "500 mg daily [metformin | dosage]"},
            default_response="Default answer",
        )
        resp1 = client.generate("Tell me the dose")
        assert "500 mg daily" in resp1.text

        resp2 = client.generate("Unknown query")
        assert resp2.text == "Default answer"


class TestGeminiClientDiskCache:
    def test_cache_miss_then_hit(self, tmp_path: Path):
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "Generated answer"

        client = GeminiClient(
            api_key="fake-key",
            cache_dir=tmp_path / "cache",
            max_retries=1,
        )
        client._model = mock_model

        # First call: cache miss
        resp1 = client.generate(prompt="What is X?")
        assert resp1.success is True
        assert resp1.text == "Generated answer"
        assert resp1.from_cache is False
        assert mock_model.generate_content.call_count == 1

        # Second call: cache hit (model should not be called again)
        resp2 = client.generate(prompt="What is X?")
        assert resp2.success is True
        assert resp2.text == "Generated answer"
        assert resp2.from_cache is True
        assert mock_model.generate_content.call_count == 1


class TestGeminiClientRetryAndFallback:
    def test_retries_on_transient_error(self, tmp_path: Path):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Success on third try"

        # Fail twice with exception, succeed on 3rd
        mock_model.generate_content.side_effect = [
            Exception("Rate limit 429"),
            Exception("Service unavailable 503"),
            mock_response,
        ]

        client = GeminiClient(
            api_key="fake-key",
            cache_dir=tmp_path / "cache",
            max_retries=3,
            initial_backoff_sec=0.001,  # fast test
        )
        client._model = mock_model

        resp = client.generate("Test prompt")
        assert resp.success is True
        assert resp.text == "Success on third try"
        assert mock_model.generate_content.call_count == 3

    def test_graceful_failure_never_crashes(self, tmp_path: Path):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Persistent API outage")

        client = GeminiClient(
            api_key="fake-key",
            cache_dir=tmp_path / "cache",
            max_retries=2,
            initial_backoff_sec=0.001,
        )
        client._model = mock_model

        # Must NOT raise exception
        resp = client.generate("Test prompt")
        assert resp.success is False
        assert resp.text == ""
        assert "Persistent API outage" in (resp.error or "")

    def test_missing_api_key_fails_gracefully(self, tmp_path: Path):
        client = GeminiClient(api_key="", cache_dir=tmp_path / "cache")
        resp = client.generate("Hello")
        assert resp.success is False
        assert "API key" in (resp.error or "")
