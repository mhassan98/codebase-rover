from __future__ import annotations

from typing import Protocol


class ModelUnavailable(RuntimeError):
    """The selected model runtime could not answer."""


class HostedModelDisabled(RuntimeError):
    """A hosted model was requested without the required opt-in."""


class ContextTooLarge(RuntimeError):
    """The prompt was larger than the context that may be sent to a model."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    provider_name: str
    model_name: str

    def complete(self, messages: list[dict[str, str]], *, context_stats: dict) -> str: ...
