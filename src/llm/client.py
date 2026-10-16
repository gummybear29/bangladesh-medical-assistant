"""Task 4 — LLM Client Interface and Google Gemini Integration.

Provides an abstract LLMClient, a production GeminiClient with SHA-256
disk caching and exponential backoff retry, and a FakeLLMClient for unit
testing and demo mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_CACHE_DIR,
    LLM_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


# ── Exceptions ─────────────────────────────────────────────────────────


class LLMServiceError(Exception):
    """Raised when the LLM service is unavailable or rate-limited."""


# ── Response Data Model ────────────────────────────────────────────────



@dataclass
class LLMResponse:
    """Unified response object returned by all LLM clients."""

    text: str
    success: bool = True
    error: str | None = None
    from_cache: bool = False
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Abstract Base Class ────────────────────────────────────────────────


class LLMClient(ABC):
    """Abstract interface for LLM text generation."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response given a prompt and optional system prompt."""
        raise NotImplementedError


# ── Fake LLM Client (Tests / Demo Mode) ────────────────────────────────


class FakeLLMClient(LLMClient):
    """Deterministic mock client for testing and offline/demo execution."""

    def __init__(
        self,
        default_response: str = "This is a mock answer based on label data.",
        responses: dict[str, str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.responses = responses or {}
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        prompt_lower = prompt.lower()
        # If this is a RAG prompt with "Question: ...", prioritize matching on the question text
        # so keywords don't accidentally match background chunks in context
        q_text = prompt_lower
        if "question:" in prompt_lower:

            parts = prompt_lower.split("question:", 1)
            if len(parts) > 1:
                q_text = parts[1].split("answer:", 1)[0].strip()

        # Check keyword mappings against question text first
        for key, ans in self.responses.items():
            if key.lower() in q_text:
                return LLMResponse(
                    text=ans,
                    success=True,
                    from_cache=False,
                    model="fake-llm",
                )

        # Fall back to full prompt matching if not matched in question
        for key, ans in self.responses.items():
            if key.lower() in prompt_lower:
                return LLMResponse(
                    text=ans,
                    success=True,
                    from_cache=False,
                    model="fake-llm",
                )


        return LLMResponse(
            text=self.default_response,
            success=True,
            from_cache=False,
            model="fake-llm",
        )


# ── Google Gemini Client ───────────────────────────────────────────────


class GeminiClient(LLMClient):
    """Production client using Google Gemini API (free tier) with caching and retries."""

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL,
        cache_dir: Path = LLM_CACHE_DIR,
        max_retries: int = LLM_MAX_RETRIES,
        initial_backoff_sec: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None  # Lazily instantiated

    def _get_cache_path(self, prompt: str, system_prompt: str | None) -> Path:
        cache_input = f"{system_prompt or ''}|||{self.model_name}|||{prompt}"
        key = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _init_gemini(self) -> None:
        if self._model is not None:
            return

        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(model_name=self.model_name)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate response with SHA-256 disk caching and exponential backoff retry.

        Never crashes: returns an LLMResponse with success=False if all retries fail.
        """
        eff_system = system if system is not None else system_prompt

        # 1. Check Disk Cache
        cache_file = self._get_cache_path(prompt, eff_system)
        if cache_file.exists():

            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                logger.debug("LLM cache hit: %s", cache_file.name)
                return LLMResponse(
                    text=data["text"],
                    success=True,
                    from_cache=True,
                    model=self.model_name,
                )
            except Exception as exc:
                logger.warning("Corrupt cache file %s: %s", cache_file.name, exc)

        # 2. Check API Key
        if not self.api_key:
            err = "Gemini API key is missing or not configured."
            logger.warning(err)
            return LLMResponse(
                text="",
                success=False,
                error=err,
                from_cache=False,
                model=self.model_name,
            )

        # 3. Assemble prompt payload
        full_content = prompt
        if eff_system:
            full_content = f"{eff_system}\n\n---\n\n{prompt}"


        # 4. Execute with retries & exponential backoff
        last_error = ""
        backoff = self.initial_backoff_sec

        for attempt in range(1, self.max_retries + 1):
            try:
                self._init_gemini()
                res = self._model.generate_content(full_content)
                text = getattr(res, "text", "") or ""

                # Write to disk cache
                try:
                    payload = {"text": text, "model": self.model_name}
                    cache_file.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception as cache_exc:
                    logger.warning("Failed to save LLM cache: %s", cache_exc)

                return LLMResponse(
                    text=text,
                    success=True,
                    from_cache=False,
                    model=self.model_name,
                )

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Gemini API attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2

        # 5. Graceful failure object
        logger.error("Gemini API calls failed after %d attempts: %s", self.max_retries, last_error)
        return LLMResponse(
            text="",
            success=False,
            error=last_error,
            from_cache=False,
            model=self.model_name,
        )


# ── Local Transformers Client (Zero API Key, 100% Offline) ─────────────


class LocalTransformersClient(LLMClient):
    """Local open-source neural LLM client using Hugging Face Transformers.
    Runs 100% locally on CPU without requiring any API key or external service.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        cache_dir: Path = LLM_CACHE_DIR,
        max_new_tokens: int = 250,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None

    def _get_cache_path(self, prompt: str, system_prompt: str | None) -> Path:
        cache_input = f"{system_prompt or ''}|||{self.model_name}|||{prompt}"
        key = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _init_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Initializing LocalTransformersClient with %s on CPU...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
        )
        logger.info("LocalTransformersClient successfully loaded on CPU.")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        # 1. Check local disk cache
        cache_file = self._get_cache_path(prompt, system_prompt)
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return LLMResponse(
                    text=data["text"],
                    success=True,
                    from_cache=True,
                    model=self.model_name,
                )
            except Exception as e:
                logger.warning("Cache read failed: %s", e)

        # 2. Run inference on CPU
        try:
            self._init_model()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer([text], return_tensors="pt")

            import torch

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
            response_text = self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0].strip()

            # 3. Write to disk cache
            try:
                payload = {"text": response_text, "model": self.model_name}
                cache_file.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as cache_exc:
                logger.warning("Failed to save LLM cache: %s", cache_exc)

            return LLMResponse(
                text=response_text,
                success=True,
                from_cache=False,
                model=self.model_name,
            )
        except Exception as exc:
            logger.error("LocalTransformersClient inference error: %s", exc)
            return LLMResponse(
                text="",
                success=False,
                error=str(exc),
                from_cache=False,
                model=self.model_name,
            )

