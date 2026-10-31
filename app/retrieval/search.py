from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import CodeChunk, Repository

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PHRASE = re.compile(r'"([^"]{2,80})"')
_STOP = {
    "the",
    "and",
    "how",
    "does",
    "what",
    "when",
    "where",
    "why",
    "who",
    "for",
    "with",
    "from",
    "this",
    "that",
    "are",
    "was",
    "were",
    "have",
    "has",
    "not",
    "can",
    "will",
    "into",
    "about",
    "code",
    "file",
    "does",
    "which",
    "should",
    "would",
    "could",
    "please",
}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: CodeChunk
    repository_name: str
    local_path: str
    provider: str


def search_chunks(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    query: str,
    repository_id: uuid.UUID | None = None,
    query_embedding: list[float] | None = None,
    limit: int = 8,
) -> list[RetrievedChunk]:
    text = " ".join(query.split())[:500]
    if not text:
        return []
    lists = [
        _text_ids(session, workspace_id, repository_id, text),
        _phrase_ids(session, workspace_id, repository_id, extract_phrases(query)),
        _identifier_ids(session, workspace_id, repository_id, extract_identifiers(query)),
    ]
    if query_embedding is not None:
        lists.append(_vector_ids(session, workspace_id, repository_id, query_embedding))
    fused = reciprocal_rank_fusion(lists)[: max(limit, 1)]
    if not fused:
        return []
    return _load(session, workspace_id, fused)


def extract_phrases(query: str) -> list[str]:
    return [match.group(1).strip() for match in _PHRASE.finditer(query) if match.group(1).strip()][:5]


def extract_identifiers(query: str) -> list[str]:
    found: list[str] = []
    for match in _IDENTIFIER.finditer(query):
        token = match.group(0)
        if token.lower() in _STOP or token in found:
            continue
        found.append(token)
        if len(found) == 8:
            break
    return found


def reciprocal_rank_fusion(rankings: list[list[uuid.UUID]], k: int = 60) -> list[uuid.UUID]:
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for index, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + index + 1)
    return [item for item, _score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]


def _scope(statement, workspace_id: uuid.UUID, repository_id: uuid.UUID | None):
    statement = statement.where(CodeChunk.workspace_id == workspace_id)
    if repository_id is not None:
        statement = statement.where(CodeChunk.repository_id == repository_id)
    return statement


def _text_ids(session: Session, workspace_id: uuid.UUID, repository_id: uuid.UUID | None, query: str) -> list[uuid.UUID]:
    document = func.to_tsvector(
        "english",
        func.concat_ws(
            " ",
            CodeChunk.file_path,
            func.coalesce(CodeChunk.symbol_name, ""),
            CodeChunk.content,
        ),
    )
    tsq = func.plainto_tsquery("english", query)
    statement = _scope(
        select(CodeChunk.id).where(document.op("@@")(tsq)).order_by(func.ts_rank(document, tsq).desc()).limit(20),
        workspace_id,
        repository_id,
    )
    return list(session.scalars(statement).all())


def _phrase_ids(
    session: Session,
    workspace_id: uuid.UUID,
    repository_id: uuid.UUID | None,
    phrases: list[str],
) -> list[uuid.UUID]:
    if not phrases:
        return []
    clauses = [CodeChunk.content.ilike(f"%{_escape_like(phrase)}%", escape="\\") for phrase in phrases]
    statement = _scope(select(CodeChunk.id).where(or_(*clauses)).limit(20), workspace_id, repository_id)
    return list(session.scalars(statement).all())


def _identifier_ids(
    session: Session,
    workspace_id: uuid.UUID,
    repository_id: uuid.UUID | None,
    identifiers: list[str],
) -> list[uuid.UUID]:
    if not identifiers:
        return []
    clauses = []
    for identifier in identifiers:
        pattern = f"%{_escape_like(identifier)}%"
        clauses.append(CodeChunk.symbol_name.ilike(pattern, escape="\\"))
        clauses.append(CodeChunk.file_path.ilike(pattern, escape="\\"))
        clauses.append(CodeChunk.content.ilike(pattern, escape="\\"))
    statement = _scope(select(CodeChunk.id).where(or_(*clauses)).limit(20), workspace_id, repository_id)
    return list(session.scalars(statement).all())


def _vector_ids(
    session: Session,
    workspace_id: uuid.UUID,
    repository_id: uuid.UUID | None,
    embedding: list[float],
) -> list[uuid.UUID]:
    statement = _scope(
        select(CodeChunk.id)
        .where(CodeChunk.embedding.is_not(None))
        .order_by(CodeChunk.embedding.cosine_distance(embedding))
        .limit(20),
        workspace_id,
        repository_id,
    )
    return list(session.scalars(statement).all())


def _load(session: Session, workspace_id: uuid.UUID, ids: list[uuid.UUID]) -> list[RetrievedChunk]:
    rows = session.execute(
        select(CodeChunk, Repository.name, Repository.local_path, Repository.provider)
        .join(Repository, Repository.id == CodeChunk.repository_id)
        .where(CodeChunk.id.in_(ids), CodeChunk.workspace_id == workspace_id)
    ).all()
    by_id = {}
    for chunk, name, local_path, provider in rows:
        if chunk.workspace_id != workspace_id:
            continue
        by_id[chunk.id] = RetrievedChunk(
            chunk=chunk,
            repository_name=name,
            local_path=local_path,
            provider=provider,
        )
    return [by_id[item] for item in ids if item in by_id]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
