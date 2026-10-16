"""Task 6 — FastAPI API Server.

Endpoints:
- GET /health: Health-check returning {"status": "ok"}.
- POST /ask: Processes questions via AnswerEngine and returns structured JSON.

Features:
- Structured JSON logging per request to logs/app.jsonl.
- Graceful error handling for LLM downtime returning HTTP 503.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.answer.engine import AnswerEngine, AnswerResult
from src.config import APP_LOG_FILE
from src.llm.client import LLMServiceError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FDA Drug Label RAG API",
    description="Question-answering on FDA-approved drug labels with mandatory citations and confidence gating.",
    version="1.0.0",
)

# ── Dependency Injection ────────────────────────────────────────────────

_engine_instance: AnswerEngine | None = None


def get_engine() -> AnswerEngine:
    """Dependency provider for AnswerEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AnswerEngine()
    return _engine_instance


# ── Pydantic Request & Response Models ─────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., description="Plain-language question about an FDA-approved drug.")


class SourceItem(BaseModel):
    drug: str
    drug_name: str
    section: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: str
    abstained: bool
    confidence_score: float
    review_queued: bool


class HealthResponse(BaseModel):
    status: str


# ── Structured JSON Logging ────────────────────────────────────────────


def log_request(
    log_file: Path,
    question: str,
    retrieved_chunk_ids: list[str],
    scores: list[float],
    latency_ms: float,
    confidence: str,
    abstained: bool,
    outcome: str,
    error: str | None = None,
) -> None:
    """Write one structured JSON line to the access log."""
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "scores": [round(s, 4) for s in scores],
            "latency_ms": round(latency_ms, 2),
            "confidence": confidence,
            "abstained": abstained,
            "outcome": outcome,
        }
        if error:
            entry["error"] = error

        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.error("Failed to write to application log %s: %s", log_file, exc)


# ── Routes ─────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    engine: AnswerEngine = Depends(get_engine),
) -> AskResponse:
    """Answer a plain-language question using FDA drug labels."""
    start_time = time.perf_counter()
    q = payload.question.strip()

    try:
        result: AnswerResult = engine.answer_question(q)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        outcome = "abstained" if result.abstained else "answered"
        # Extract chunk provenance
        chunk_ids = result.retrieved_chunk_ids or [f"{s.drug}__{s.section}" for s in result.sources]
        scores = [result.confidence_score] if result.confidence_score else []


        log_request(
            log_file=APP_LOG_FILE,
            question=q,
            retrieved_chunk_ids=chunk_ids,
            scores=scores,
            latency_ms=latency_ms,
            confidence=result.confidence,
            abstained=result.abstained,
            outcome=outcome,
        )

        return AskResponse(
            answer=result.answer,
            sources=[
                SourceItem(
                    drug=s.drug,
                    drug_name=s.drug_name,
                    section=s.section,
                    url=s.url,
                )
                for s in result.sources
            ],
            confidence=result.confidence,
            abstained=result.abstained,
            confidence_score=round(result.confidence_score, 4),
            review_queued=result.review_queued,
        )

    except LLMServiceError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_request(
            log_file=APP_LOG_FILE,
            question=q,
            retrieved_chunk_ids=[],
            scores=[],
            latency_ms=latency_ms,
            confidence="low",
            abstained=True,
            outcome="error",
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {str(exc)}",
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.exception("Unexpected error in /ask: %s", exc)
        log_request(
            log_file=APP_LOG_FILE,
            question=q,
            retrieved_chunk_ids=[],
            scores=[],
            latency_ms=latency_ms,
            confidence="low",
            abstained=True,
            outcome="error",
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing the request.",
        )
