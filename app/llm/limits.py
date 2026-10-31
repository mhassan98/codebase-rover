from __future__ import annotations

from app.config import Settings
from app.llm.base import ContextTooLarge


def assert_prompt_within_limits(messages: list[dict[str, str]], context_stats: dict, settings: Settings) -> None:
    chunks = int(context_stats.get("chunks") or 0)
    chars = int(context_stats.get("chars") or 0)
    if chunks > settings.max_context_chunks:
        raise ContextTooLarge("retrieved context exceeds the chunk limit")
    if chars > settings.max_retrieved_chars + settings.max_diff_prompt_chars:
        raise ContextTooLarge("retrieved context exceeds the character limit")
    total = sum(len(str(message.get("content") or "")) for message in messages)
    if total > settings.max_prompt_chars:
        raise ContextTooLarge("prompt exceeds the maximum size sent to a model")
