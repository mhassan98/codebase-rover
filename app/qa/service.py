from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.llm.base import ChatModel, ContextTooLarge, ModelUnavailable
from app.llm.context import build_qa_messages
from app.llm.factory import select_chat_provider
from app.llm.jsonutil import parse_json_object
from app.llm.ollama import OllamaEmbedder
from app.models import Question, Workspace
from app.retrieval.search import search_chunks
from app.retrieval.source import VerifiedExcerpt, verify_retrieved

log = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = (
    "Insufficient evidence in this workspace to answer that from the indexed source."
)
MODEL_UNAVAILABLE = (
    "The language model is unavailable, so there is no synthesized answer. "
    "Matching source is listed in related excerpts and is not a confirmed explanation."
)


@dataclass
class AnswerDraft:
    answer: str
    sufficient: bool
    synthesis: str
    citations: list[VerifiedExcerpt]
    related_excerpts: list[VerifiedExcerpt]
    provider_name: str | None = None
    model_name: str | None = None


def synthesize_answer(
    question: str,
    excerpts: list[VerifiedExcerpt],
    provider: ChatModel | None,
    settings: Settings,
) -> AnswerDraft:
    provider_name = None if provider is None else provider.provider_name
    model_name = None if provider is None else provider.model_name
    if not excerpts:
        return AnswerDraft(INSUFFICIENT_EVIDENCE, False, "not_run", [], [], provider_name, model_name)
    if provider is None:
        return AnswerDraft(MODEL_UNAVAILABLE, False, "unavailable", [], excerpts, None, None)
    bundle = build_qa_messages(question, excerpts, settings)
    try:
        raw = provider.complete(bundle.messages, context_stats=bundle.stats)
        payload = parse_json_object(raw)
    except (ModelUnavailable, ContextTooLarge, ValueError, TypeError):
        log.info("question synthesis unavailable")
        return AnswerDraft(MODEL_UNAVAILABLE, False, "unavailable", [], excerpts, provider_name, model_name)
    return _from_payload(payload, excerpts, provider_name, model_name)


def answer_question(
    session: Session,
    workspace: Workspace,
    question: str,
    repository_id: uuid.UUID | None,
    provider: ChatModel,
    settings: Settings | None = None,
    embedder=None,
) -> Question:
    settings = settings or get_settings()
    embedding = _embed_query(embedder or OllamaEmbedder(settings), question)
    retrieved = search_chunks(
        session,
        workspace_id=workspace.id,
        query=question,
        repository_id=repository_id,
        query_embedding=embedding,
        limit=settings.max_context_chunks,
    )
    excerpts = verify_retrieved(retrieved)
    draft = synthesize_answer(question, excerpts, provider, settings)
    row = Question(
        workspace_id=workspace.id,
        repository_id=repository_id,
        question_text=question,
        answer_text=draft.answer,
        sufficient=draft.sufficient,
        synthesis=draft.synthesis,
        citations_json={
            "citations": [_excerpt_dict(item) for item in draft.citations],
            "related_excerpts": [_excerpt_dict(item) for item in draft.related_excerpts],
        },
        model_provider=draft.provider_name,
        model_name=draft.model_name,
    )
    session.add(row)
    session.commit()
    return row


def enforce_chat_provider(settings: Settings, preference: str, confirm: bool) -> ChatModel:
    return select_chat_provider(settings, preference, confirm)


def _from_payload(payload: dict, excerpts: list[VerifiedExcerpt], provider_name: str, model_name: str) -> AnswerDraft:
    by_id = {item.citation_id: item for item in excerpts}
    raw_ids = payload.get("citation_ids")
    ids = raw_ids if isinstance(raw_ids, list) else []
    citations = [by_id[item] for item in ids if isinstance(item, str) and item in by_id]
    answer = payload.get("answer") if isinstance(payload.get("answer"), str) else ""
    sufficient = bool(payload.get("sufficient")) and bool(citations) and bool(answer.strip())
    if not sufficient:
        return AnswerDraft(INSUFFICIENT_EVIDENCE, False, "model", [], excerpts, provider_name, model_name)
    return AnswerDraft(answer.strip()[:16000], True, "model", citations, excerpts, provider_name, model_name)


def _embed_query(embedder, question: str) -> list[float] | None:
    try:
        vectors = embedder.embed([question[:500]])
    except (ModelUnavailable, OSError, ValueError):
        log.info("query embedding unavailable; using text search")
        return None
    if not vectors:
        return None
    return vectors[0]


def _excerpt_dict(excerpt: VerifiedExcerpt) -> dict:
    return asdict(excerpt)
