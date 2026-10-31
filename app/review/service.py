from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict
from pathlib import Path

from sqlalchemy.orm import Session

from app.catalog import repository_in_workspace, utcnow
from app.config import Settings, get_settings
from app.connectors.base import GitError, ProviderCapabilityError
from app.connectors.registry import connector_for_repository
from app.llm.base import ChatModel, ContextTooLarge, ModelUnavailable
from app.llm.context import build_review_messages
from app.llm.jsonutil import parse_json_object
from app.llm.ollama import OllamaEmbedder
from app.models import ReviewJob, Workspace
from app.qa.service import _embed_query
from app.retrieval.search import extract_identifiers, search_chunks
from app.retrieval.source import VerifiedExcerpt, verify_retrieved
from app.review.diffparse import DiffFile, DiffParseError, parse_unified_diff, summarize_files
from app.review.validate import Finding, ReviewAssessment, validate_model_review

log = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

MODEL_UNAVAILABLE_SUMMARY = (
    "The language model is unavailable, so no findings were produced. "
    "The parsed diff is evidence of what changed and is not a defect list."
)


def run_review(
    session: Session,
    workspace: Workspace,
    *,
    diff_text: str | None,
    head_sha: str | None,
    base_sha: str | None,
    repository_id: uuid.UUID | None,
    title: str | None,
    provider: ChatModel,
    settings: Settings | None = None,
    embedder=None,
) -> ReviewJob:
    settings = settings or get_settings()
    if diff_text is not None and len(diff_text) > settings.max_diff_chars:
        raise DiffParseError("diff exceeds the configured size limit")
    repository = None
    connector = None
    resolved_head = None
    resolved_base = None
    verified = False
    if repository_id is not None:
        repository = repository_in_workspace(session, workspace, repository_id)
        if repository is None:
            raise GitError("repository is not in this workspace")
        connector = connector_for_repository(repository.provider, repository.local_path)
        if not head_sha:
            raise GitError("head_sha is required when a repository is selected")
        resolved_head = connector.resolve_commit(head_sha)
        if base_sha:
            resolved_base = connector.resolve_commit(base_sha)
        verified = True
    if not diff_text:
        if connector is None or resolved_head is None or resolved_base is None:
            raise GitError("provide a diff, or a repository with base_sha and head_sha")
        diff_text = connector.diff(resolved_base, resolved_head)
    if head_sha and resolved_head is None:
        resolved_head = head_sha.strip()
    submitted_head = head_sha
    try:
        diff_files = parse_unified_diff(diff_text)
    except DiffParseError:
        raise
    head_texts: dict[str, str] = {}
    head_excerpts: list[VerifiedExcerpt] = []
    limitations: list[str] = []
    if connector is not None and resolved_head is not None:
        head_texts, head_excerpts, read_limits = _read_head(connector, resolved_head, diff_files, repository.name)
        limitations.extend(read_limits)
        if base_sha and resolved_base and diff_text:
            limitations.extend(_diff_mismatch_note(connector, resolved_base, resolved_head, diff_files))
    elif repository_id is None:
        limitations.append("No repository was selected, so the head revision was not read from git.")
    related = _related_excerpts(session, workspace, diff_files, settings, embedder or OllamaEmbedder(settings))
    assessment = synthesize_review(
        diff_text,
        diff_files,
        head_texts,
        head_excerpts + related,
        provider,
        settings,
    )
    assessment.limitations = _unique(limitations + assessment.limitations)
    context_excerpts = [_excerpt_dict(item) for item in head_excerpts + related]
    row = ReviewJob(
        workspace_id=workspace.id,
        repository_id=repository.id if repository is not None else None,
        title=(title or "")[:200] or None,
        head_sha=resolved_head,
        submitted_head=submitted_head,
        base_sha=resolved_base or base_sha,
        diff_text=diff_text,
        status="succeeded",
        synthesis=assessment.synthesis,
        verified_against_git=verified,
        summary=assessment.summary,
        result_json=_result_payload(assessment, diff_files, context_excerpts),
        model_provider=provider.provider_name,
        model_name=provider.model_name,
        error=None,
        finished_at=utcnow(),
    )
    session.add(row)
    session.commit()
    return row


def synthesize_review(
    diff_text: str,
    diff_files: list[DiffFile],
    head_texts: dict[str, str],
    excerpts: list[VerifiedExcerpt],
    provider: ChatModel | None,
    settings: Settings,
) -> ReviewAssessment:
    if not diff_text.strip() or not diff_files:
        return ReviewAssessment(
            summary="The diff is empty.",
            limitations=["The diff is empty."],
            synthesis="not_run",
        )
    bundle = build_review_messages(diff_text, excerpts, settings)
    limitations = []
    if bundle.diff_truncated:
        limitations.append("The diff was truncated before it was sent to the model.")
    if provider is None:
        return ReviewAssessment(
            summary=MODEL_UNAVAILABLE_SUMMARY,
            limitations=limitations,
            synthesis="unavailable",
        )
    try:
        raw = provider.complete(bundle.messages, context_stats=bundle.stats)
        payload = parse_json_object(raw)
    except (ModelUnavailable, ContextTooLarge, ValueError, TypeError):
        log.info("review synthesis unavailable")
        return ReviewAssessment(summary=MODEL_UNAVAILABLE_SUMMARY, limitations=limitations, synthesis="unavailable")
    assessment = validate_model_review(payload, diff_files, diff_text, head_texts)
    assessment.limitations = _unique(limitations + assessment.limitations)
    assessment.synthesis = "model"
    if not assessment.summary:
        assessment.summary = "Review completed from the diff and retrieved source."
    return assessment


def publish_review_comments(session: Session, workspace: Workspace, review: ReviewJob) -> None:
    if review.repository_id is None:
        raise ProviderCapabilityError(
            "This review is not attached to a repository, so there is no provider to publish comments to."
        )
    repository = repository_in_workspace(session, workspace, review.repository_id)
    if repository is None:
        raise ProviderCapabilityError("repository is not in this workspace")
    connector = connector_for_repository(repository.provider, repository.local_path)
    from app.connectors.base import ReviewCommentDraft, ReviewDraft

    comments = []
    for finding in review.result_json.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        comments.append(
            ReviewCommentDraft(
                file_path=str(finding.get("file_path") or ""),
                line=int(finding.get("end_line") or finding.get("start_line") or 1),
                body=str(finding.get("explanation") or finding.get("title") or ""),
            )
        )
    connector.publish_review(str(review.id), ReviewDraft(summary=review.summary, comments=comments))


def _related_excerpts(session, workspace, diff_files, settings: Settings, embedder) -> list[VerifiedExcerpt]:
    query = _review_query(diff_files)
    if not query.strip():
        return []
    embedding = _embed_query(embedder, query)
    retrieved = search_chunks(
        session,
        workspace_id=workspace.id,
        query=query,
        query_embedding=embedding,
        limit=settings.max_context_chunks,
    )
    changed = {item.path for item in diff_files}
    preferred = [item for item in retrieved if item.chunk.file_path not in changed]
    chosen = preferred or retrieved
    excerpts = verify_retrieved(chosen)
    for index, excerpt in enumerate(excerpts, start=1):
        excerpt_id = f"r{index}"
        excerpts[index - 1] = VerifiedExcerpt(
            citation_id=excerpt_id,
            repository_id=excerpt.repository_id,
            repository_name=excerpt.repository_name,
            file_path=excerpt.file_path,
            start_line=excerpt.start_line,
            end_line=excerpt.end_line,
            commit_sha=excerpt.commit_sha,
            text=excerpt.text,
            index_drift=excerpt.index_drift,
        )
    return excerpts


def _read_head(connector, head_sha: str, diff_files: list[DiffFile], repository_name: str):
    texts: dict[str, str] = {}
    excerpts: list[VerifiedExcerpt] = []
    limitations: list[str] = []
    for item in diff_files:
        path = item.new_path
        if item.binary:
            limitations.append(f"{item.path} is binary and was not read.")
            continue
        if not path:
            limitations.append(f"{item.path or 'a file'} is deleted at the head revision.")
            continue
        try:
            text = connector.read_text(head_sha, path)
        except GitError:
            limitations.append(f"Could not read {path} at the head revision.")
            continue
        texts[path] = text
        numbered = _numbered_hunk_excerpt(text, item)
        if numbered is None:
            continue
        body, start_line, end_line = numbered
        excerpts.append(
            VerifiedExcerpt(
                citation_id=f"h{len(excerpts) + 1}",
                repository_id="",
                repository_name=repository_name,
                file_path=path,
                start_line=start_line,
                end_line=end_line,
                commit_sha=head_sha,
                text=body,
                index_drift=False,
            )
        )
    return texts, excerpts, limitations


def _numbered_hunk_excerpt(text: str, diff_file: DiffFile, margin: int = 3):
    lines = text.splitlines()
    ranges: list[tuple[int, int]] = []
    for hunk in diff_file.hunks:
        if hunk.new_count <= 0:
            continue
        start = max(1, hunk.new_start - margin)
        end = min(len(lines), hunk.new_start + max(hunk.new_count, 1) - 1 + margin)
        if end >= start:
            ranges.append((start, end))
    if not ranges:
        return None
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    parts: list[str] = []
    for start, end in merged:
        for number in range(start, end + 1):
            parts.append(f"{number}|{lines[number - 1]}")
    return "\n".join(parts), merged[0][0], merged[-1][1]


def _diff_mismatch_note(connector, base_sha: str, head_sha: str, submitted: list[DiffFile]) -> list[str]:
    try:
        computed = parse_unified_diff(connector.diff(base_sha, head_sha))
    except (GitError, DiffParseError):
        return []
    submitted_paths = {item.path for item in submitted}
    computed_paths = {item.path for item in computed}
    if submitted_paths != computed_paths:
        return ["The submitted diff file set differs from git diff of the recorded base and head commits."]
    return []


def _review_query(diff_files: list[DiffFile]) -> str:
    tokens: list[str] = []
    for item in diff_files:
        stem = Path(item.path).stem
        if stem and stem not in {".", ""}:
            tokens.append(stem)
        for hunk in item.hunks:
            for line in hunk.lines:
                if line.kind != "add":
                    continue
                tokens.extend(match.group(0) for match in _IDENTIFIER.finditer(line.text))
    unique: list[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
        if len(unique) == 20:
            break
    identifiers = extract_identifiers(" ".join(unique))
    return " ".join(identifiers or unique[:12])


def _excerpt_dict(excerpt: VerifiedExcerpt) -> dict:
    return asdict(excerpt)


def _result_payload(
    assessment: ReviewAssessment,
    diff_files: list[DiffFile],
    context_excerpts: list[dict],
) -> dict:
    return {
        "summary": assessment.summary,
        "findings": [asdict(item) if isinstance(item, Finding) else item for item in assessment.findings],
        "assumptions": assessment.assumptions,
        "limitations": assessment.limitations,
        "rejected": assessment.rejected,
        "diff_files": summarize_files(diff_files),
        "context_excerpts": context_excerpts,
    }


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen[:20]
