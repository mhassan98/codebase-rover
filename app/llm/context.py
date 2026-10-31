from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import Settings
from app.retrieval.source import VerifiedExcerpt

QA_SYSTEM = (
    "You answer questions about one software workspace using only the source excerpts provided. "
    "Do not describe behavior that is not supported by those excerpts. "
    "If the excerpts are not enough, set sufficient to false and leave citation_ids empty. "
    "When sufficient is true, citation_ids must list only excerpt ids that support the answer. "
    'Respond with one JSON object: {"sufficient": boolean, "answer": string, "citation_ids": [string]}.'
)

REVIEW_SYSTEM = (
    "You review a code change using the diff, exact source at the head revision, and related excerpts. "
    "Report a finding only when the diff or the head source shows the issue. "
    "Each finding must quote evidence from the diff or the head source and use line numbers on the changed lines. "
    "Do not present speculation as a finding. Put guesses and unverified risks in assumptions. "
    "Do not call an issue confirmed unless the diff or head source shows it directly. "
    "If context is missing, list that in limitations instead of inventing behavior. "
    "Respond with one JSON object with keys summary, findings, assumptions, and limitations. "
    "Each finding has severity (critical, high, medium, or low), confidence (confirmed or likely), "
    "title, explanation, file_path, start_line, end_line, and evidence (array of exact quotes)."
)


@dataclass(frozen=True)
class PromptBundle:
    messages: list[dict[str, str]]
    stats: dict
    diff_truncated: bool = False


def build_qa_messages(question: str, excerpts: list[VerifiedExcerpt], settings: Settings) -> PromptBundle:
    chosen, stats = trim_excerpts(excerpts, settings.max_context_chunks, settings.max_retrieved_chars)
    blocks = [_format_excerpt(item) for item in chosen]
    body = "Question:\n" + question.strip() + "\n\nExcerpts:\n"
    body += "\n\n".join(blocks) if blocks else "(none)"
    return PromptBundle(
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": body},
        ],
        stats=stats,
    )


def build_review_messages(
    diff_text: str,
    excerpts: list[VerifiedExcerpt],
    settings: Settings,
) -> PromptBundle:
    shown_diff, truncated = truncate_text(diff_text, settings.max_diff_prompt_chars)
    chosen, excerpt_stats = trim_excerpts(excerpts, settings.max_context_chunks, settings.max_retrieved_chars)
    head = [item for item in chosen if item.citation_id.startswith("h")]
    related = [item for item in chosen if not item.citation_id.startswith("h")]
    parts = ["Diff:", shown_diff or "(empty)", "", "Head revision excerpts:"]
    parts.append("\n\n".join(_format_excerpt(item) for item in head) if head else "(none)")
    parts.extend(["", "Related workspace excerpts:"])
    parts.append("\n\n".join(_format_excerpt(item) for item in related) if related else "(none)")
    stats = {
        "chunks": excerpt_stats["chunks"],
        "chars": excerpt_stats["chars"] + len(shown_diff),
        "truncated": excerpt_stats["truncated"] or truncated,
    }
    return PromptBundle(
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM},
            {"role": "user", "content": "\n".join(parts)},
        ],
        stats=stats,
        diff_truncated=truncated,
    )


def trim_excerpts(excerpts: list[VerifiedExcerpt], max_chunks: int, max_chars: int) -> tuple[list[VerifiedExcerpt], dict]:
    chosen: list[VerifiedExcerpt] = []
    total = 0
    truncated = False
    for excerpt in excerpts:
        if len(chosen) >= max_chunks:
            truncated = True
            break
        remaining = max_chars - total
        if remaining <= 0:
            truncated = True
            break
        text = excerpt.text
        if len(text) > remaining:
            text = text[:remaining]
            truncated = True
        total += len(text)
        chosen.append(excerpt if text == excerpt.text else replace(excerpt, text=text))
        if len(text) < len(excerpt.text):
            break
    if len(chosen) < len(excerpts):
        truncated = True
    return chosen, {"chunks": len(chosen), "chars": total, "truncated": truncated}


def truncate_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit] + "\n\n[diff truncated]\n", True


def _format_excerpt(excerpt: VerifiedExcerpt) -> str:
    return (
        f"[{excerpt.citation_id}] repository={excerpt.repository_name} path={excerpt.file_path} "
        f"lines={excerpt.start_line}-{excerpt.end_line} commit={excerpt.commit_sha}\n{excerpt.text}"
    )
