"""Tests for Task 6 — FastAPI Endpoints (/ask and /health).

Verifies:
1. GET /health returns 200 and {"status": "ok"}.
2. POST /ask returns structured JSON with answer, sources, confidence, and abstained flag.
3. POST /ask abstains gracefully when confidence is low.
4. Structured JSON logging is appended to logs/app.jsonl with latency, retrieved chunk IDs, and outcome.
5. LLM service errors return HTTP 503 with a user-friendly error message.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.answer.engine import AnswerEngine, AnswerResult, SourceRef
from src.api.app import app, get_engine
from src.config import DISCLAIMER_TEXT
from src.llm.client import FakeLLMClient, LLMServiceError
from src.retrieval.hybrid import RetrievedChunk


def _make_dummy_chunk(
    chunk_id: str = "metformin__dosage__0",
    drug_name: str = "metformin",
    section: str = "dosage",
    score: float = 0.032,
    similarity: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Starting dose is 500 mg.",
        drug_name=drug_name,
        section=section,
        source_url="https://dailymed.nlm.nih.gov/test-metformin",
        chunk_index=0,
        score=score,
        similarity=similarity,
        vector_similarity=similarity,
    )


@pytest.fixture
def client_and_engine(tmp_path: Path):
    """Provide a TestClient with a controlled AnswerEngine and isolated log file."""
    log_file = tmp_path / "app.jsonl"
    queue_file = tmp_path / "review_queue.jsonl"

    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_dummy_chunk()]
    llm = FakeLLMClient(default_response="Metformin starting dose is 500 mg [metformin | dosage].")

    engine = AnswerEngine(
        retriever=retriever,
        llm_client=llm,
        review_queue_file=queue_file,
    )

    app.dependency_overrides[get_engine] = lambda: engine

    with patch("src.api.app.APP_LOG_FILE", log_file):
        test_client = TestClient(app)
        yield test_client, engine, log_file

    app.dependency_overrides.clear()


def test_health_check(client_and_engine):
    client, _, _ = client_and_engine
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ask_endpoint_happy_path(client_and_engine):
    client, _, log_file = client_and_engine
    payload = {"question": "What is the starting dose of metformin?"}
    res = client.post("/ask", json=payload)

    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "500 mg" in data["answer"]
    assert DISCLAIMER_TEXT in data["answer"]
    assert data["confidence"] == "high"
    assert data["abstained"] is False
    assert len(data["sources"]) == 1
    assert data["sources"][0]["drug"] == "metformin"
    assert data["sources"][0]["section"] == "dosage"

    # Verify structured JSON logging
    assert log_file.exists()
    logs = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(logs) == 1
    log_entry = logs[0]
    assert log_entry["question"] == "What is the starting dose of metformin?"
    assert log_entry["outcome"] == "answered"
    assert log_entry["retrieved_chunk_ids"] == ["metformin__dosage__0"]
    assert "latency_ms" in log_entry
    assert log_entry["latency_ms"] >= 0


def test_ask_endpoint_abstention(client_and_engine):
    client, engine, log_file = client_and_engine
    # Force low score below threshold
    engine.retriever.retrieve.return_value = [_make_dummy_chunk(score=0.002, similarity=0.60)]

    res = client.post("/ask", json={"question": "Unknown compound dose?"})
    assert res.status_code == 200
    data = res.json()
    assert data["abstained"] is True
    assert data["confidence"] == "low"
    assert "Insufficient evidence" in data["answer"]
    assert data["review_queued"] is True

    # Verify structured log recorded abstained outcome
    logs = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
    assert logs[-1]["outcome"] == "abstained"
    assert logs[-1]["abstained"] is True


def test_ask_endpoint_llm_failure_returns_503(client_and_engine):
    client, engine, log_file = client_and_engine

    def fail_generate(*args, **kwargs):
        raise LLMServiceError("Service unavailable (503 overloaded)")

    engine.llm_client.generate = fail_generate

    res = client.post("/ask", json={"question": "What is the dose?"})
    assert res.status_code == 503
    data = res.json()
    assert "detail" in data
    assert "LLM service unavailable" in data["detail"]

    # Verify structured log recorded error outcome
    logs = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
    assert logs[-1]["outcome"] == "error"
    assert "error" in logs[-1]


def test_ask_endpoint_empty_question(client_and_engine):
    client, _, _ = client_and_engine
    res = client.post("/ask", json={"question": "   "})
    assert res.status_code == 200
    data = res.json()
    assert data["abstained"] is True
    assert "Insufficient evidence" in data["answer"]
