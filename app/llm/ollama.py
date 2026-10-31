from __future__ import annotations

import httpx

from app.config import Settings
from app.indexing.limits import EMBEDDING_DIM
from app.llm.base import ModelUnavailable
from app.llm.limits import assert_prompt_within_limits


class OllamaEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.ollama_embed_model
        self.base = settings.ollama_base_url.rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        timeout = httpx.Timeout(self.settings.embed_timeout_seconds, connect=3.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.base}/api/embed",
                    json={"model": self.model, "input": texts},
                )
                if response.status_code == 404:
                    vectors = [self._embed_one(client, text) for text in texts]
                elif response.status_code >= 400:
                    raise ModelUnavailable(self._missing_message())
                else:
                    payload = response.json()
                    vectors = payload.get("embeddings")
                    if not isinstance(vectors, list):
                        raise ModelUnavailable("Ollama returned an unexpected embedding payload")
        except ModelUnavailable:
            raise
        except (httpx.HTTPError, ValueError):
            raise ModelUnavailable(self._missing_message()) from None
        self._check_dimensions(vectors)
        return vectors

    def _embed_one(self, client: httpx.Client, text: str) -> list[float]:
        response = client.post(
            f"{self.base}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        if response.status_code >= 400:
            raise ModelUnavailable(self._missing_message())
        vector = response.json().get("embedding")
        if not isinstance(vector, list):
            raise ModelUnavailable("Ollama returned an unexpected embedding payload")
        return vector

    def _check_dimensions(self, vectors: list) -> None:
        if len(vectors) == 0:
            return
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
                found = len(vector) if isinstance(vector, list) else 0
                raise ModelUnavailable(
                    f"embedding dimension {found} does not match the schema width {EMBEDDING_DIM}"
                )

    def _missing_message(self) -> str:
        return (
            "Ollama embeddings are unavailable. Pull the embedding model with: "
            f"ollama pull {self.model}"
        )


class OllamaChatProvider:
    provider_name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.ollama_chat_model
        self.base = settings.ollama_base_url.rstrip("/")

    def complete(self, messages: list[dict[str, str]], *, context_stats: dict) -> str:
        assert_prompt_within_limits(messages, context_stats, self.settings)
        try:
            return self._chat(messages, force_json=True)
        except ModelUnavailable:
            return self._chat(messages, force_json=False)

    def _chat(self, messages: list[dict[str, str]], *, force_json: bool) -> str:
        body: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if force_json:
            body["format"] = "json"
        timeout = httpx.Timeout(self.settings.chat_timeout_seconds, connect=3.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{self.base}/api/chat", json=body)
        except httpx.HTTPError:
            raise ModelUnavailable(self._missing_message()) from None
        if response.status_code >= 400:
            raise ModelUnavailable(self._missing_message())
        try:
            content = response.json().get("message", {}).get("content", "")
        except ValueError:
            raise ModelUnavailable(self._missing_message()) from None
        if not isinstance(content, str) or not content.strip():
            raise ModelUnavailable(self._missing_message())
        return content

    def _missing_message(self) -> str:
        return (
            "Ollama chat is unavailable. Pull the chat model with: "
            f"ollama pull {self.model_name}"
        )
