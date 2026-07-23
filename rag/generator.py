"""LLM Generator Module.

Flexible LLM generation layer supporting API providers (OpenAI, Groq) with an intelligent local fallback.
"""

import os
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load .env from project root automatically
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
    logger_init = logging.getLogger(__name__)
    logger_init.info(f"Loaded .env from {_env_path}")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMGenerator:
    """Interface for generating answers via API or local fallback."""

    def __init__(self, model_provider: Optional[str] = None):
        """Initializes generator checking for available API keys."""
        raw_openai = os.getenv("OPENAI_API_KEY", "").strip()
        raw_groq = os.getenv("GROQ_API_KEY", "").strip()

        # Only treat a key as valid if it has the correct prefix — avoids picking up
        # stale placeholder keys from other projects in the environment.
        # Accept both classic (sk-) and project-scoped (sk-proj-) OpenAI API keys
        self.openai_key = raw_openai if (raw_openai.startswith("sk-proj-") or raw_openai.startswith("sk-")) else None
        self.groq_key = raw_groq if raw_groq.startswith("gsk_") else None

        self.provider = model_provider or self._detect_provider()
        logger.info(f"Initialized LLM Generator with active provider: '{self.provider}'")

    def _detect_provider(self) -> str:
        if self.groq_key:
            return "groq"
        elif self.openai_key:
            return "openai"
        else:
            return "local_fallback"

    def generate(self, system_prompt: str, user_prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generates grounded answer using configured provider."""
        if self.provider == "groq":
            return self._generate_groq(system_prompt, user_prompt, retrieved_chunks)
        elif self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, retrieved_chunks)
        else:
            return self._generate_local_fallback(user_prompt, retrieved_chunks)

    def _generate_groq(self, system_prompt: str, user_prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generates answer using Groq API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                logger.warning(f"Groq API returned error status {resp.status_code}. Using local fallback.")
                return self._generate_local_fallback(user_prompt, retrieved_chunks)
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return self._generate_local_fallback(user_prompt, retrieved_chunks)

    def _generate_openai(self, system_prompt: str, user_prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generates answer using OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                logger.warning(f"OpenAI API returned error status {resp.status_code}. Using local fallback.")
                return self._generate_local_fallback(user_prompt, retrieved_chunks)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._generate_local_fallback(user_prompt, retrieved_chunks)

    def _generate_local_fallback(self, user_prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Synthesizes grounded extractive answer directly from top retrieved context chunks."""
        if not retrieved_chunks:
            return "Based on the provided document, I cannot find sufficient information to answer your question."

        # Build a clean multi-chunk extractive summary
        answer_parts = []
        for i, chunk in enumerate(retrieved_chunks[:3], start=1):
            source = chunk.get("source", "Document")
            page = chunk.get("page", 1)
            text = chunk.get("text", "").strip()
            # Take first 3 clean lines from each chunk
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            extract = " ".join(lines[:3])
            if extract:
                answer_parts.append(f"**From {source} (Page {page}):** {extract}")

        if not answer_parts:
            return "Based on the provided document, I cannot find sufficient information to answer your question."

        return "\n\n".join(answer_parts)


# Provider priority: Groq (free + fast) > OpenAI > local extractive fallback