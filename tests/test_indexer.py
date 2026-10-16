"""Tests for src.ingest.indexer — Task 2.

All embedding model inferences and heavy downloads are mocked for fast,
deterministic unit testing.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.indexer import (
    Chunk,
    build_bm25_index,
    chunk_drug_label,
    chunk_section,
    clean_label_text,
    count_tokens,
    index_labels,
    split_into_sentences,
    tokenize_for_bm25,
    upsert_to_chroma,
)


# ── Fixtures & Mocks ───────────────────────────────────────────────────


class MockEmbeddingModel:
    """Mock sentence-transformers model returning deterministic vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        sentences: list[str] | str,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> list[list[float]]:
        if isinstance(sentences, str):
            sentences = [sentences]
        # Return a simple deterministic vector based on text length
        return [[float(len(s) % 10)] * self.dim for s in sentences]


@pytest.fixture
def mock_embedder() -> MockEmbeddingModel:
    return MockEmbeddingModel()


@pytest.fixture
def sample_label_dict() -> dict:
    return {
        "drug_name": "metformin",
        "set_id": "test-set-id",
        "effective_time": "20240101",
        "sections": {
            "indications": (
                "Metformin hydrochloride tablets are indicated as an adjunct to diet "
                "and exercise to improve glycemic control in adults with type 2 diabetes mellitus."
            ),
            "dosage": (
                "The recommended starting dose of metformin hydrochloride tablets is 500 mg orally "
                "twice a day or 850 mg once a day, given with meals."
            ),
            "warnings": "Lactic acidosis is a rare, but serious, complication of metformin therapy.",
            "interactions": None,
            "contraindications": "Severe renal impairment (eGFR below 30 mL/min/1.73 m2).",
        },
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=test-set-id",
    }


# ── Token counting & sentence splitting ───────────────────────────────


class TestTokenAndSentenceUtils:
    def test_count_tokens_returns_positive_integer(self):
        text = "Metformin is used for diabetes."
        count = count_tokens(text)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_string(self):
        assert count_tokens("") == 0

    def test_split_into_sentences_basic(self):
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = split_into_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"

    def test_split_into_sentences_empty(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   \n\t  ") == []


# ── Text Cleaning ─────────────────────────────────────────────────────


class TestCleanLabelText:
    def test_strips_bracketed_cross_references(self):
        text = "Risk of myopathy increases [ see Warnings and Precautions (5.1) ] in older patients."
        cleaned = clean_label_text(text)
        assert cleaned == "Risk of myopathy increases in older patients."

    def test_strips_parenthetical_section_citations(self):
        text = "Initial dose is 10 mg ( 2.1 ). Multiple sections apply ( 4 , 5.5 , 7.1 )."
        cleaned = clean_label_text(text)
        assert cleaned == "Initial dose is 10 mg. Multiple sections apply."

    def test_preserves_clinical_numbers(self):
        text = "Take 500 mg daily for 10 days. eGFR < 30 mL/min."
        cleaned = clean_label_text(text)
        assert cleaned == "Take 500 mg daily for 10 days. eGFR < 30 mL/min."


# ── Chunking logic ────────────────────────────────────────────────────


class TestChunkSection:
    def test_chunk_empty_or_none(self):
        assert chunk_section("aspirin", "dosage", None, "http://url") == []
        assert chunk_section("aspirin", "dosage", "", "http://url") == []
        assert chunk_section("aspirin", "dosage", "   ", "http://url") == []

    def test_chunk_short_text_single_chunk(self):
        text = "Take 500 mg daily with water."
        chunks = chunk_section("aspirin", "dosage", text, "http://example.com/aspirin")
        assert len(chunks) == 1
        c = chunks[0]
        assert c.chunk_id == "aspirin__dosage__0"
        assert c.drug_name == "aspirin"
        assert c.section == "dosage"
        assert c.chunk_index == 0
        assert c.source_url == "http://example.com/aspirin"
        assert c.text == f"Drug: aspirin | Section: dosage\n\n{text}"

    def test_chunk_prepends_drug_and_section_header(self):
        text = "Take 1 tablet daily."
        chunks = chunk_section("lisinopril", "dosage", text, "http://lisinopril")
        assert len(chunks) == 1
        assert chunks[0].text.startswith("Drug: lisinopril | Section: dosage\n\n")
        assert chunks[0].text == f"Drug: lisinopril | Section: dosage\n\n{text}"

    def test_chunk_long_text_multiple_chunks_with_overlap(self):
        # Create text with many sentences (> 500 tokens)
        sentences = [
            f"Sentence number {i} provides clinical details about drug safety and dosing recommendations in patients."
            for i in range(50)
        ]
        text = " ".join(sentences)
        total_tokens = count_tokens(text)
        assert total_tokens > 500

        chunks = chunk_section(
            "atorvastatin",
            "warnings",
            text,
            "http://example.com/atorvastatin",
            target_tokens=100,
            overlap_tokens=20,
        )
        assert len(chunks) > 1

        for i, c in enumerate(chunks):
            assert c.chunk_id == f"atorvastatin__warnings__{i}"
            assert c.chunk_index == i
            assert c.drug_name == "atorvastatin"
            assert c.section == "warnings"
            assert count_tokens(c.text) <= 150  # within sensible bounds of target

    def test_chunk_single_very_long_sentence(self):
        # A single sentence with 200 repeated words without punctuation
        long_sentence = "word " * 300
        chunks = chunk_section(
            "drugx",
            "indications",
            long_sentence,
            "http://example.com",
            target_tokens=100,
            overlap_tokens=20,
        )
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text.strip()) > 0

    def test_chunk_deterministic_ids(self):
        text = "Sample sentence one. Sample sentence two."
        chunks1 = chunk_section("metformin", "indications", text, "http://url")
        chunks2 = chunk_section("metformin", "indications", text, "http://url")
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


# ── Label file chunking ───────────────────────────────────────────────


class TestChunkDrugLabel:
    def test_chunk_drug_label_skips_none_sections(self, sample_label_dict: dict):
        chunks = chunk_drug_label(sample_label_dict)
        # sample_label_dict has 4 populated sections: indications, dosage, warnings, contraindications
        sections_in_chunks = {c.section for c in chunks}
        assert "interactions" not in sections_in_chunks
        assert sections_in_chunks == {"indications", "dosage", "warnings", "contraindications"}

        for c in chunks:
            assert c.drug_name == "metformin"
            assert c.source_url == sample_label_dict["source_url"]
            assert c.chunk_id.startswith("metformin__")


# ── BM25 Indexing ─────────────────────────────────────────────────────


class TestBM25Index:
    def test_tokenize_for_bm25(self):
        tokens = tokenize_for_bm25("Metformin HCl, 500 mg: take orally!")
        assert tokens == ["metformin", "hcl", "500", "mg", "take", "orally"]

    def test_build_and_query_bm25_index(self, tmp_path: Path):
        chunks = [
            Chunk(
                chunk_id="metformin__indications__0",
                text="Metformin is indicated for glycemic control in type 2 diabetes mellitus.",
                drug_name="metformin",
                section="indications",
                source_url="http://metformin",
                chunk_index=0,
            ),
            Chunk(
                chunk_id="atorvastatin__indications__0",
                text="Atorvastatin is indicated to reduce LDL cholesterol and cardiovascular risk.",
                drug_name="atorvastatin",
                section="indications",
                source_url="http://atorvastatin",
                chunk_index=0,
            ),
            Chunk(
                chunk_id="amoxicillin__indications__0",
                text="Amoxicillin is indicated for the treatment of bacterial infections.",
                drug_name="amoxicillin",
                section="indications",
                source_url="http://amoxicillin",
                chunk_index=0,
            ),
        ]

        index_file = tmp_path / "bm25_index.pkl"
        bm25, chunk_map = build_bm25_index(chunks, save_path=index_file)

        assert index_file.exists()
        assert len(chunk_map) == 3

        # Verify pickled payload can be reloaded
        with open(index_file, "rb") as f:
            data = pickle.load(f)
            assert "bm25" in data
            assert "chunk_map" in data
            assert len(data["chunk_map"]) == 3

        # Verify search query
        tokenized_query = tokenize_for_bm25("diabetes")
        scores = bm25.get_scores(tokenized_query)
        assert scores[0] > 0
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]


# ── ChromaDB Upsert (Mocked Embeddings) ───────────────────────────────


class TestChromaUpsert:
    def test_upsert_to_chroma(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        chroma_dir = tmp_path / "chroma_test"
        collection_name = "test_collection"

        chunks = [
            Chunk(
                chunk_id="metformin__indications__0",
                text="Metformin indication text.",
                drug_name="metformin",
                section="indications",
                source_url="http://metformin",
                chunk_index=0,
            ),
            Chunk(
                chunk_id="metformin__dosage__0",
                text="Metformin dosage text.",
                drug_name="metformin",
                section="dosage",
                source_url="http://metformin",
                chunk_index=0,
            ),
        ]

        collection = upsert_to_chroma(
            chunks=chunks,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model=mock_embedder,
        )

        assert collection.count() == 2

        # Verify retrieved item from Chroma
        result = collection.get(ids=["metformin__indications__0"], include=["metadatas", "documents"])
        assert len(result["ids"]) == 1
        assert result["documents"][0] == "Metformin indication text."
        assert result["metadatas"][0]["drug_name"] == "metformin"
        assert result["metadatas"][0]["section"] == "indications"
        assert result["metadatas"][0]["source_url"] == "http://metformin"


# ── Full Indexing Pipeline ────────────────────────────────────────────


class TestIndexLabelsPipeline:
    def test_index_labels_end_to_end(
        self,
        tmp_path: Path,
        sample_label_dict: dict,
        mock_embedder: MockEmbeddingModel,
    ):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "metformin.json").write_text(
            json.dumps(sample_label_dict), encoding="utf-8"
        )

        chroma_dir = tmp_path / "chroma"
        bm25_file = tmp_path / "bm25.pkl"

        summary = index_labels(
            labels_dir=labels_dir,
            chroma_dir=chroma_dir,
            collection_name="test_pipeline",
            bm25_path=bm25_file,
            embedding_model=mock_embedder,
        )

        assert summary["total_drugs"] == 1
        assert summary["total_chunks"] == 4
        assert bm25_file.exists()
        assert chroma_dir.exists()
