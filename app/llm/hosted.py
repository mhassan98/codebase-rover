from __future__ import annotations

import httpx

from app.config import Settings
from app.llm.base import HostedModelDisabled, ModelUnavailable
from app.llm.limits import assert_prompt_within_limits


class HostedChatProvider:
    """OpenAI-compatible chat. Embeddings stay on the local runtime.

    The caller must already have dropped everything except the question or diff
    and the retrieved excerpts. This provider sends that message list and nothing else.
    """

    provider_name = "hosted"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        if not settings.hosted_model_enabled:
            raise HostedModelDisabled("hosted models are disabled")
        if not settings.hosted_model_base_url or not settings.hosted_model_name:
            raise HostedModelDisabled("hosted model URL and name are required")
        self.settings = settings
        self.model_name = settings.hosted_model_name
        self._client = client

    def complete(self, messages: list[dict[str, str]], *, context_stats: dict) -> str:
        assert_prompt_within_limits(messages, context_stats, self.settings)
        url = self.settings.hosted_model_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = self.settings.hosted_model_api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": self.model_name,
            "temperature": 0.1,
            "messages": messages,
        }
        owns_client = self._client is None
        timeout = httpx.Timeout(self.settings.chat_timeout_seconds, connect=3.0)
        client = self._client or httpx.Client(timeout=timeout)
        try:
            response = client.post(url, json=body, headers=headers)
        except httpx.HTTPError:
            raise ModelUnavailable("hosted model request failed") from None
        finally:
            if owns_client:
                client.close()
        if response.status_code >= 400:
            raise ModelUnavailable("hosted model request failed")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelUnavailable("hosted model returned an unexpected response") from None
        if not isinstance(content, str) or not content.strip():
            raise ModelUnavailable("hosted model returned an empty response")
        return content
