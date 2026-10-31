import json

import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.llm.base import ContextTooLarge, HostedModelDisabled
from app.llm.context import build_qa_messages
from app.llm.factory import models_payload, select_chat_provider
from app.llm.hosted import HostedChatProvider
from app.redact import redact
from app.retrieval.source import VerifiedExcerpt


def _settings(**overrides) -> Settings:
    base = dict(
        hosted_model_enabled=False,
        hosted_model_base_url="https://models.example.test/v1",
        hosted_model_name="frontier",
        hosted_model_api_key=SecretStr("sk-test-secret"),
        max_context_chunks=8,
        max_retrieved_chars=5000,
        max_diff_prompt_chars=1000,
        max_prompt_chars=8000,
    )
    base.update(overrides)
    return Settings(**base)


def _excerpt(citation_id: str, text: str) -> VerifiedExcerpt:
    return VerifiedExcerpt(
        citation_id=citation_id,
        repository_id="repo",
        repository_name="fastcrib",
        file_path="src/tax.py",
        start_line=1,
        end_line=2,
        commit_sha="abc123",
        text=text,
        index_drift=False,
    )


def test_hosted_model_is_off_until_enabled_and_confirmed() -> None:
    with pytest.raises(HostedModelDisabled):
        select_chat_provider(_settings(), "hosted", True)
    enabled = _settings(hosted_model_enabled=True)
    with pytest.raises(HostedModelDisabled):
        select_chat_provider(enabled, "hosted", False)
    provider = select_chat_provider(enabled, "hosted", True)
    assert provider.provider_name == "hosted"
    assert provider.model_name == "frontier"


def test_model_description_hides_secret() -> None:
    payload = models_payload(_settings(hosted_model_enabled=True))
    encoded = json.dumps(payload)
    assert "sk-test-secret" not in encoded
    assert "https://models.example.test" not in encoded
    assert payload["hosted"]["enabled"] is True
    assert payload["hosted"]["api_key_configured"] is True
    assert payload["default_provider"] == "ollama"


def test_prompt_contains_only_selected_excerpts() -> None:
    excerpts = [_excerpt(f"c{index}", f"SENTINEL{index} " + ("x" * 1000)) for index in range(12)]
    bundle = build_qa_messages("Where is tax calculated?", excerpts, _settings())
    body = bundle.messages[1]["content"]
    assert "SENTINEL0" in body
    assert "SENTINEL8" not in body
    assert bundle.stats["chunks"] <= 8
    assert bundle.stats["chars"] <= 5000
    assert "UNRELATED_SECRET" not in body


def test_hosted_completion_posts_only_the_messages() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"sufficient\": false}"}}]})

    provider = HostedChatProvider(_settings(hosted_model_enabled=True), client=httpx.Client(transport=httpx.MockTransport(handler)))
    messages = [
        {"role": "system", "content": "answer from excerpts"},
        {"role": "user", "content": "question and one excerpt"},
    ]
    result = provider.complete(messages, context_stats={"chunks": 1, "chars": 24})
    assert "sufficient" in result
    assert captured["body"]["messages"] == messages
    assert set(captured["body"]) == {"model", "temperature", "messages"}
    assert captured["authorization"] == "Bearer sk-test-secret"
    assert "sk-test-secret" not in captured["url"]


def test_oversized_prompt_is_not_sent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("hosted model was called")

    provider = HostedChatProvider(
        _settings(hosted_model_enabled=True),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ContextTooLarge):
        provider.complete(
            [{"role": "user", "content": "x" * 9000}],
            context_stats={"chunks": 1, "chars": 10},
        )


def test_redact_secrets() -> None:
    assert "sk-live" not in redact("Authorization: Bearer sk-live")
    assert "hunter2" not in redact("api_key=hunter2")
