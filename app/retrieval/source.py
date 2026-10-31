from __future__ import annotations

from dataclasses import dataclass

from app.connectors.base import GitError
from app.connectors.registry import connector_for_repository
from app.retrieval.search import RetrievedChunk


@dataclass(frozen=True)
class VerifiedExcerpt:
    citation_id: str
    repository_id: str
    repository_name: str
    file_path: str
    start_line: int
    end_line: int
    commit_sha: str
    text: str
    index_drift: bool


def slice_lines(text: str, start_line: int, end_line: int) -> str | None:
    lines = text.splitlines()
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        return None
    return "\n".join(lines[start_line - 1 : end_line])


def verify_retrieved(items: list[RetrievedChunk]) -> list[VerifiedExcerpt]:
    """Replace index text with the exact git revision before it can be cited."""
    connectors = {}
    files: dict[tuple[str, str, str], str] = {}
    excerpts: list[VerifiedExcerpt] = []
    for item in items:
        key_root = item.local_path
        try:
            connector = connectors.get(key_root)
            if connector is None:
                connector = connector_for_repository(item.provider, item.local_path)
                connectors[key_root] = connector
            cache_key = (item.local_path, item.chunk.commit_sha, item.chunk.file_path)
            if cache_key not in files:
                files[cache_key] = connector.read_text(item.chunk.commit_sha, item.chunk.file_path)
            snippet = slice_lines(files[cache_key], item.chunk.start_line, item.chunk.end_line)
        except GitError:
            continue
        if snippet is None:
            continue
        excerpts.append(
            VerifiedExcerpt(
                citation_id=f"c{len(excerpts) + 1}",
                repository_id=str(item.chunk.repository_id),
                repository_name=item.repository_name,
                file_path=item.chunk.file_path,
                start_line=item.chunk.start_line,
                end_line=item.chunk.end_line,
                commit_sha=item.chunk.commit_sha,
                text=snippet,
                index_drift=snippet != item.chunk.content,
            )
        )
    return excerpts
