"""Task 3 — Hybrid Retrieval (Vector + BM25 + Reciprocal Rank Fusion).

Combines dense vector search via ChromaDB and sparse lexical search via
rank-bm25, merging candidate rankings using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from src.config import (
    BM25_INDEX_FILE,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    RRF_K,
)
from src.ingest.indexer import get_embedding_model, tokenize_for_bm25

logger = logging.getLogger(__name__)


# ── Data Model ─────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """A retrieved chunk enriched with hybrid ranking scores and provenance."""

    chunk_id: str
    text: str
    drug_name: str
    section: str
    source_url: str
    chunk_index: int
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── RRF Scoring ────────────────────────────────────────────────────────


def compute_rrf_score(
    vector_rank: int | None,
    bm25_rank: int | None,
    k: int = RRF_K,
) -> float:
    """Calculate the Reciprocal Rank Fusion score for an item across two legs.

    Formula: score(d) = sum(1 / (k + rank_i)) for each leg in which d appears.
    Ranks are 1-based (1, 2, ...).
    """
    score = 0.0
    if vector_rank is not None and vector_rank > 0:
        score += 1.0 / (k + vector_rank)
    if bm25_rank is not None and bm25_rank > 0:
        score += 1.0 / (k + bm25_rank)
    return score


# ── Hybrid Retriever ───────────────────────────────────────────────────


class HybridRetriever:
    """Coordinates dense vector and sparse BM25 retrieval with RRF fusion."""

    def __init__(
        self,
        collection: Any,
        bm25: BM25Okapi,
        chunk_map: list[dict[str, Any]],
        embedding_model: Any = None,
        rrf_k: int = RRF_K,
        candidate_pool: int = 20,
    ) -> None:
        self.collection = collection
        self.bm25 = bm25
        self.chunk_map = chunk_map
        self.embedding_model = embedding_model or get_embedding_model()
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self._chunk_by_id: dict[str, dict[str, Any]] = {
            c["chunk_id"]: c for c in chunk_map
        }

    @classmethod
    def from_defaults(
        cls,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION,
        bm25_path: Path = BM25_INDEX_FILE,
        embedding_model: Any = None,
        rrf_k: int = RRF_K,
    ) -> HybridRetriever:
        """Instantiate a HybridRetriever using project default persistence paths."""
        logger.info("Initializing HybridRetriever from %s and %s", chroma_dir, bm25_path)

        # 1. Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(name=collection_name)

        # 2. Load pickled BM25 index
        with open(bm25_path, "rb") as f:
            bm25_payload = pickle.load(f)
        bm25 = bm25_payload["bm25"]
        chunk_map = bm25_payload["chunk_map"]

        # 3. Load embedding model
        model = embedding_model or get_embedding_model(EMBEDDING_MODEL)

        return cls(
            collection=collection,
            bm25=bm25,
            chunk_map=chunk_map,
            embedding_model=model,
            rrf_k=rrf_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        filter_drug: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k chunks using hybrid vector + BM25 search merged with RRF.

        Args:
            query: Plain-language user question.
            top_k: Number of final chunks to return (default 5).
            filter_drug: Optional drug name to constrain retrieval.

        Returns:
            List of RetrievedChunk objects ordered by descending RRF score.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        n_candidates = max(self.candidate_pool, top_k * 2)

        # ── 1. Vector Search Leg ──────────────────────────────────────
        vector_ranks: dict[str, int] = {}
        try:
            raw_emb = self.embedding_model.encode([clean_query])
            q_emb = (
                raw_emb.tolist()
                if hasattr(raw_emb, "tolist")
                else [list(e) for e in raw_emb]
            )

            total_docs = (
                self.collection.count()
                if hasattr(self.collection, "count")
                else len(self.chunk_map)
            )
            n_results = min(n_candidates, total_docs)

            where_clause = (
                {"drug_name": filter_drug.lower()} if filter_drug else None
            )

            v_res = self.collection.query(
                query_embeddings=q_emb,
                n_results=n_results,
                where=where_clause,
            )

            if v_res and "ids" in v_res and v_res["ids"]:
                for rank, cid in enumerate(v_res["ids"][0], 1):
                    vector_ranks[cid] = rank

        except Exception as exc:
            logger.error("Vector search failed during hybrid retrieval: %s", exc)

        # ── 2. BM25 Search Leg ────────────────────────────────────────
        bm25_ranks: dict[str, int] = {}
        try:
            tokens = tokenize_for_bm25(clean_query)
            if tokens:
                bm25_scores = self.bm25.get_scores(tokens)

                # Identify eligible candidate indices
                if filter_drug:
                    target_drug = filter_drug.lower()
                    eligible_indices = [
                        i
                        for i, c in enumerate(self.chunk_map)
                        if c.get("drug_name", "").lower() == target_drug
                    ]
                else:
                    eligible_indices = list(range(len(self.chunk_map)))

                # Sort eligible candidates by BM25 score
                sorted_indices = sorted(
                    eligible_indices,
                    key=lambda idx: bm25_scores[idx],
                    reverse=True,
                )

                # Take top candidates with positive score
                for rank, idx in enumerate(sorted_indices[:n_candidates], 1):
                    if bm25_scores[idx] > 0 or len(bm25_ranks) < top_k:
                        cid = self.chunk_map[idx]["chunk_id"]
                        bm25_ranks[cid] = rank

        except Exception as exc:
            logger.error("BM25 search failed during hybrid retrieval: %s", exc)

        # ── 3. Reciprocal Rank Fusion ─────────────────────────────────
        candidate_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        if filter_drug:
            target = filter_drug.lower()
            candidate_ids = {
                cid
                for cid in candidate_ids
                if self._chunk_by_id.get(cid, {}).get("drug_name", "").lower() == target
            }

        if not candidate_ids:
            logger.warning("No candidate chunks found for query: %s", clean_query)
            return []

        scored_candidates: list[tuple[str, float]] = []
        for cid in candidate_ids:
            score = compute_rrf_score(
                vector_rank=vector_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                k=self.rrf_k,
            )
            scored_candidates.append((cid, score))

        # Sort descending by RRF score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # ── 4. Build Output Models ────────────────────────────────────
        results: list[RetrievedChunk] = []
        for cid, score in scored_candidates[:top_k]:
            meta = self._chunk_by_id.get(cid)
            if not meta:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=meta["text"],
                    drug_name=meta["drug_name"],
                    section=meta["section"],
                    source_url=meta["source_url"],
                    chunk_index=meta["chunk_index"],
                    score=score,
                    vector_rank=vector_ranks.get(cid),
                    bm25_rank=bm25_ranks.get(cid),
                )
            )

        return results
