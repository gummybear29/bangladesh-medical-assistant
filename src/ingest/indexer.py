"""Task 2 — Chunking & Embedding Pipeline.

Splits drug label sections into ~400-token chunks with sentence-aware
boundary detection and overlap, generates embeddings, upserts into ChromaDB,
and builds an in-process BM25 index.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
import tiktoken
from rank_bm25 import BM25Okapi

from src.config import (
    BM25_INDEX_FILE,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBEDDING_MODEL,
    LABELS_DIR,
)

logger = logging.getLogger(__name__)

# Lazily initialized tiktoken encoding
_ENCODER = None


def get_token_encoder() -> tiktoken.Encoding:
    """Return the cached cl100k_base tiktoken encoder."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken."""
    if not text:
        return 0
    return len(get_token_encoder().encode(text))


# ── Data Model ─────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """Represents a chunk of text with its associated label metadata."""

    chunk_id: str
    text: str
    drug_name: str
    section: str
    source_url: str
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Text Splitting ─────────────────────────────────────────────────────


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common boundaries.

    Handles periods, exclamation marks, question marks, and line breaks.
    """
    if not text or not text.strip():
        return []

    # Split by standard sentence terminators followed by whitespace, or newlines
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def _split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Split a single sentence exceeding max_tokens into smaller word pieces."""
    words = sentence.split()
    parts: list[str] = []
    current_words: list[str] = []

    for word in words:
        current_words.append(word)
        candidate = " ".join(current_words)
        if count_tokens(candidate) >= max_tokens:
            parts.append(candidate)
            current_words = []

    if current_words:
        parts.append(" ".join(current_words))

    return parts


def chunk_section(
    drug_name: str,
    section: str,
    text: str | None,
    source_url: str,
    target_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split a label section's text into overlapping chunks of ~target_tokens.

    Args:
        drug_name: Name of the drug (e.g. 'metformin').
        section: Section name (e.g. 'indications').
        text: Section content text (or None).
        source_url: Deep-link URL for attribution.
        target_tokens: Desired max tokens per chunk.
        overlap_tokens: Desired token overlap between consecutive chunks.

    Returns:
        List of Chunk objects with deterministic IDs.
    """
    if not text or not text.strip():
        return []

    raw_sentences = split_into_sentences(text)
    if not raw_sentences:
        return []

    # Normalize sentences: break any unusually long single sentence
    sentences: list[str] = []
    for s in raw_sentences:
        if count_tokens(s) > target_tokens:
            sentences.extend(_split_long_sentence(s, target_tokens))
        else:
            sentences.append(s)

    chunks: list[Chunk] = []
    chunk_index = 0
    start_idx = 0
    n = len(sentences)

    while start_idx < n:
        current_sentences: list[str] = []
        current_tokens = 0
        end_idx = start_idx

        while end_idx < n:
            s = sentences[end_idx]
            s_tokens = count_tokens(s)
            if current_sentences and (current_tokens + s_tokens > target_tokens):
                break
            current_sentences.append(s)
            current_tokens += s_tokens
            end_idx += 1

        body_text = " ".join(current_sentences)
        chunk_text = f"Drug: {drug_name} | Section: {section}\n\n{body_text}"
        chunk_id = f"{drug_name}__{section}__{chunk_index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                drug_name=drug_name,
                section=section,
                source_url=source_url,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

        if end_idx >= n:
            break

        # Calculate overlap for next chunk start
        overlap_accum = 0
        overlap_idx = end_idx
        while overlap_idx > start_idx:
            overlap_accum += count_tokens(sentences[overlap_idx - 1])
            if overlap_accum >= overlap_tokens:
                break
            overlap_idx -= 1

        # Prevent infinite loops: next start must advance past previous start
        start_idx = max(overlap_idx, start_idx + 1)

    return chunks


def chunk_drug_label(
    label_data: dict[str, Any],
    target_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Extract and chunk all populated sections from a drug label JSON dict."""
    drug_name = label_data["drug_name"]
    source_url = label_data.get("source_url", "")
    sections = label_data.get("sections", {})

    all_chunks: list[Chunk] = []
    for section_name, section_text in sections.items():
        if section_text:
            chunks = chunk_section(
                drug_name=drug_name,
                section=section_name,
                text=section_text,
                source_url=source_url,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
            all_chunks.extend(chunks)

    return all_chunks


# ── BM25 Indexing ─────────────────────────────────────────────────────


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase words/tokens for BM25 matching.

    Strips punctuation and extracts alphanumeric word tokens.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def build_bm25_index(
    chunks: list[Chunk],
    save_path: Path = BM25_INDEX_FILE,
) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Build a BM25Okapi index over chunk texts and pickle it to disk.

    Returns:
        Tuple of (BM25Okapi instance, list of chunk metadata dicts).
    """
    tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    chunk_map = [c.to_dict() for c in chunks]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_map": chunk_map}, f)

    logger.info("Saved BM25 index with %d chunks to %s", len(chunks), save_path)
    return bm25, chunk_map


# ── ChromaDB Upsert ───────────────────────────────────────────────────


def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s ...", model_name)
    return SentenceTransformer(model_name)


def upsert_to_chroma(
    chunks: list[Chunk],
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    embedding_model: Any = None,
    batch_size: int = 128,
) -> chromadb.Collection:
    """Upsert chunk texts and metadata into a persistent ChromaDB collection.

    Args:
        chunks: List of Chunk objects to store.
        chroma_dir: Persistent storage directory for ChromaDB.
        collection_name: Name of the collection.
        embedding_model: SentenceTransformer model instance (or Mock). If None,
                         loads EMBEDDING_MODEL.
        batch_size: Batch size for generating embeddings and upserting.

    Returns:
        The updated chromadb Collection instance.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)

    if not chunks:
        logger.warning("No chunks to upsert into Chroma.")
        return collection

    if embedding_model is None:
        embedding_model = get_embedding_model()

    logger.info("Upserting %d chunks into Chroma collection '%s'...", len(chunks), collection_name)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [
            {
                "drug_name": c.drug_name,
                "section": c.section,
                "source_url": c.source_url,
                "chunk_index": c.chunk_index,
            }
            for c in batch
        ]

        # Generate embeddings via model
        raw_embeddings = embedding_model.encode(texts, show_progress_bar=False)
        # Ensure standard python floats
        embeddings = [
            e.tolist() if hasattr(e, "tolist") else list(e)
            for e in raw_embeddings
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    logger.info("Upsert complete. Collection count: %d", collection.count())
    return collection


# ── Full Indexing Pipeline ────────────────────────────────────────────


def index_labels(
    labels_dir: Path = LABELS_DIR,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    bm25_path: Path = BM25_INDEX_FILE,
    embedding_model: Any = None,
) -> dict[str, int]:
    """Scan data/labels/*.json, chunk all sections, upsert to Chroma, and build BM25.

    Returns:
        Summary dict with 'total_drugs' and 'total_chunks'.
    """
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        logger.warning("No label files found in %s", labels_dir)
        return {"total_drugs": 0, "total_chunks": 0}

    logger.info("Found %d label files in %s", len(label_files), labels_dir)
    all_chunks: list[Chunk] = []

    for fpath in label_files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            chunks = chunk_drug_label(data)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("Failed to parse/chunk %s: %s", fpath.name, exc)

    logger.info("Total chunks generated: %d from %d drugs", len(all_chunks), len(label_files))

    # 1. Upsert to ChromaDB
    upsert_to_chroma(
        chunks=all_chunks,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # 2. Build and save BM25 index
    build_bm25_index(
        chunks=all_chunks,
        save_path=bm25_path,
    )

    return {
        "total_drugs": len(label_files),
        "total_chunks": len(all_chunks),
    }


# ── CLI Entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    index_labels()
