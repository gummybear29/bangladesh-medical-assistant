"""Tests for src.retrieval.hybrid — Task 3.

Includes isolated unit tests (with mock Chroma and BM25) and integration
tests against the indexed drug label database.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import BM25_INDEX_FILE, CHROMA_COLLECTION, CHROMA_DIR, RRF_K
from src.ingest.indexer import Chunk
from src.retrieval.hybrid import (
    HybridRetriever,
    RetrievedChunk,
    compute_rrf_score,
    detect_drugs_in_query,
)


# ── RRF Scoring Unit Tests ─────────────────────────────────────────────


class TestRrfScoring:
    def test_compute_rrf_both_legs(self):
        # Rank 1 in vector, rank 1 in BM25 with k=60
        score = compute_rrf_score(vector_rank=1, bm25_rank=1, k=60)
        expected = (1.0 / 61) + (1.0 / 61)
        assert pytest.approx(score) == expected

    def test_compute_rrf_vector_only(self):
        score = compute_rrf_score(vector_rank=1, bm25_rank=None, k=60)
        expected = 1.0 / 61
        assert pytest.approx(score) == expected

    def test_compute_rrf_bm25_only(self):
        score = compute_rrf_score(vector_rank=None, bm25_rank=2, k=60)
        expected = 1.0 / 62
        assert pytest.approx(score) == expected

    def test_compute_rrf_neither(self):
        assert compute_rrf_score(vector_rank=None, bm25_rank=None, k=60) == 0.0

    def test_dual_leg_outranks_single_leg(self):
        # Item appearing moderately in both legs outranks item appearing high in only one
        dual_score = compute_rrf_score(vector_rank=5, bm25_rank=5, k=60)
        single_score = compute_rrf_score(vector_rank=1, bm25_rank=None, k=60)
        assert dual_score > single_score


# ── Mocked HybridRetriever Tests ──────────────────────────────────────


class MockEmbeddingModel:
    def encode(self, texts: list[str] | str, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1] * 384 for _ in texts]


@pytest.fixture
def mock_retriever():
    mock_collection = MagicMock()
    mock_bm25 = MagicMock()

    chunk_map = [
        {
            "chunk_id": "metformin__dosage__0",
            "text": "Drug: metformin | Section: dosage\n\nStarting dose is 500 mg.",
            "drug_name": "metformin",
            "section": "dosage",
            "source_url": "http://metformin",
            "chunk_index": 0,
        },
        {
            "chunk_id": "metformin__indications__0",
            "text": "Drug: metformin | Section: indications\n\nIndicated for type 2 diabetes.",
            "drug_name": "metformin",
            "section": "indications",
            "source_url": "http://metformin",
            "chunk_index": 0,
        },
        {
            "chunk_id": "atorvastatin__dosage__0",
            "text": "Drug: atorvastatin | Section: dosage\n\nStarting dose 10-20 mg.",
            "drug_name": "atorvastatin",
            "section": "dosage",
            "source_url": "http://atorvastatin",
            "chunk_index": 0,
        },
    ]

    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "ids": [["metformin__dosage__0", "atorvastatin__dosage__0"]]
    }
    mock_bm25.get_scores.return_value = [10.0, 5.0, 1.0]

    return HybridRetriever(
        collection=mock_collection,
        bm25=mock_bm25,
        chunk_map=chunk_map,
        embedding_model=MockEmbeddingModel(),
        rrf_k=60,
    )


class TestHybridRetrieverUnit:
    def test_retrieve_empty_query_returns_empty(self, mock_retriever: HybridRetriever):
        assert mock_retriever.retrieve("") == []
        assert mock_retriever.retrieve("   ") == []

    def test_retrieve_returns_top_k(self, mock_retriever: HybridRetriever):
        results = mock_retriever.retrieve("metformin dose", top_k=2)
        assert len(results) <= 2
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].drug_name == "metformin"
        assert results[0].score > 0

    def test_retrieve_preserves_metadata(self, mock_retriever: HybridRetriever):
        results = mock_retriever.retrieve("metformin dose", top_k=1)
        r = results[0]
        assert r.chunk_id == "metformin__dosage__0"
        assert r.section == "dosage"
        assert r.source_url == "http://metformin"
        assert r.vector_rank == 1
        assert r.bm25_rank == 1

    def test_retrieve_with_drug_filter(self, mock_retriever: HybridRetriever):
        results = mock_retriever.retrieve("dose", top_k=5, filter_drug="atorvastatin")
        for r in results:
            assert r.drug_name == "atorvastatin"

    def test_retrieve_auto_applies_single_drug(self, mock_retriever: HybridRetriever):
        # Query mentioning 'metformin' should auto-filter to metformin
        results = mock_retriever.retrieve("What is the dose for metformin?", top_k=5)
        for r in results:
            assert r.drug_name == "metformin"

    def test_retrieve_multi_drug_merging(self, mock_retriever: HybridRetriever):
        # Query mentioning two drugs (metformin and atorvastatin)
        results = mock_retriever.retrieve("Can I take metformin with atorvastatin?", top_k=4)
        drug_names = {r.drug_name for r in results}
        assert "metformin" in drug_names
        assert "atorvastatin" in drug_names


# ── Drug Detection Unit Tests ─────────────────────────────────────────


class TestDrugDetection:
    def test_detect_single_generic(self):
        detected = detect_drugs_in_query("Can I take metformin twice daily?")
        assert detected == ["metformin"]

    def test_detect_brand_alias(self):
        detected = detect_drugs_in_query("Is Tylenol safe in pregnancy?")
        assert detected == ["acetaminophen"]

    def test_detect_multiple_drugs(self):
        detected = detect_drugs_in_query("Can I take Advil with warfarin?")
        assert "ibuprofen" in detected
        assert "warfarin" in detected

    def test_detect_no_drug(self):
        detected = detect_drugs_in_query("What are common antibiotic interactions?")
        assert detected == []


# ── Integration Test against Live Index ───────────────────────────────


class TestHybridRetrieverIntegration:
    @pytest.mark.skipif(
        not (CHROMA_DIR.exists() and BM25_INDEX_FILE.exists()),
        reason="Chroma directory and BM25 index file must exist for integration test",
    )
    def test_metformin_maximum_dose_retrieval(self):
        """User requirement: 'What is the maximum dose of metformin?'

        must return metformin dosage chunks at rank 1-3.
        """
        retriever = HybridRetriever.from_defaults()
        query = "What is the maximum dose of metformin?"
        results = retriever.retrieve(query, top_k=5)

        assert len(results) >= 3

        # Top result must be metformin
        assert results[0].drug_name == "metformin"

        # Within the top 3 ranks, at least one chunk must be metformin dosage
        top_3_metformin_dosage = [
            r
            for r in results[:3]
            if r.drug_name == "metformin" and r.section == "dosage"
        ]
        assert len(top_3_metformin_dosage) >= 1, (
            f"Expected metformin dosage chunk in top 3, but got: "
            f"{[(r.drug_name, r.section, r.score) for r in results[:3]]}"
        )
