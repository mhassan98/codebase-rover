from __future__ import annotations

from app.config import Settings
from app.llm.base import ChatModel, HostedModelDisabled
from app.llm.hosted import HostedChatProvider
from app.llm.ollama import OllamaChatProvider, OllamaEmbedder


def select_chat_provider(settings: Settings, preference: str, confirm: bool) -> ChatModel:
    pref = (preference or "ollama").strip().lower()
    if pref == "hosted":
        if not settings.hosted_model_enabled or not confirm:
            raise HostedModelDisabled(
                "Hosted models are disabled by default. Set HOSTED_MODEL_ENABLED=true and "
                "resubmit with confirm_send_context=true. Only the question or diff and the "
                "retrieved excerpts are sent, not the repository."
            )
        if not settings.hosted_model_base_url.strip() or not settings.hosted_model_name.strip():
            raise HostedModelDisabled(
                "HOSTED_MODEL_BASE_URL and HOSTED_MODEL_NAME are required when hosted models are enabled."
            )
        return HostedChatProvider(settings)
    if pref == "ollama":
        return OllamaChatProvider(settings)
    raise ValueError(f"unknown model preference: {preference}")


def models_payload(settings: Settings) -> dict:
    secret = settings.hosted_model_api_key.get_secret_value()
    return {
        "default_provider": "ollama",
        "embedding_dimension": settings.embedding_dim,
        "ollama": {
            "base_url": settings.ollama_base_url,
            "chat_model": settings.ollama_chat_model,
            "embed_model": settings.ollama_embed_model,
        },
        "hosted": {
            "enabled": settings.hosted_model_enabled,
            "model": settings.hosted_model_name if settings.hosted_model_enabled else "",
            "base_url_configured": bool(settings.hosted_model_base_url.strip()),
            "api_key_configured": bool(secret),
            "requires": [
                "HOSTED_MODEL_ENABLED=true",
                "confirm_send_context=true on the request",
            ],
            "sends": "only the question or diff and the retrieved excerpts, not the repository",
        },
    }
