from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.indexing.limits import STRUCTURAL_LINE_LIMIT, WINDOW_LINES, WINDOW_OVERLAP

try:
    from tree_sitter_language_pack import process as _process_source
    from tree_sitter_language_pack.options import ProcessConfig
except ImportError:  # pragma: no cover
    _process_source = None
    ProcessConfig = None

EXT_LANG = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
}


@dataclass(frozen=True)
class Chunk:
    start_line: int
    end_line: int
    content: str
    language: str | None
    symbol_name: str | None
    chunk_kind: str


def language_for_path(path: str) -> str | None:
    return EXT_LANG.get(Path(path).suffix.lower())


def chunk_source(path: str, text: str) -> list[Chunk]:
    """Split source into citation-sized chunks with 1-based inclusive line ranges.

    Tree-sitter structure is used when a grammar is available. Lines outside
    those symbols fall back to windows so the file remains searchable.
    """
    lines = text.splitlines()
    if not lines:
        return []
    language = language_for_path(path)
    covered = [False] * (len(lines) + 1)
    structural = _structural_chunks(lines, language, covered)
    windows = _fill_gaps(lines, covered, language)
    chunks = structural + windows
    chunks.sort(key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.chunk_kind))
    return chunks


def _structural_chunks(lines: list[str], language: str | None, covered: list[bool]) -> list[Chunk]:
    if not language or _process_source is None or ProcessConfig is None:
        return []
    try:
        result = _process_source(
            "\n".join(lines) + "\n",
            ProcessConfig(
                language=language,
                structure=True,
                symbols=False,
                imports=False,
                exports=False,
                comments=False,
                docstrings=False,
                diagnostics=False,
                data_extraction=False,
            ),
        )
    except Exception:
        return []
    chunks: list[Chunk] = []
    _walk_structure(list(result.structure or []), lines, language, chunks, covered)
    return chunks


def _walk_structure(items, lines: list[str], language: str | None, out: list[Chunk], covered: list[bool]) -> None:
    for item in items:
        start = _clamp_line(item.span.start_line + 1, len(lines))
        end = _clamp_line(item.span.end_line + 1, len(lines))
        if end < start:
            continue
        start = _attach_leading_decorators(lines, start)
        symbol = (getattr(item, "name", None) or "").strip()[:200] or None
        kind = _kind_label(getattr(item, "kind", "block"))
        children = list(getattr(item, "children", None) or [])
        if children and kind == "function" and (end - start + 1) <= STRUCTURAL_LINE_LIMIT:
            _append_span(out, covered, lines, start, end, language, symbol, kind)
            continue
        if children:
            child_starts = []
            for child in children:
                child_start = _clamp_line(child.span.start_line + 1, len(lines))
                child_starts.append(_attach_leading_decorators(lines, child_start))
            header_end = min(child_starts) - 1
            if header_end >= start:
                _append_span(out, covered, lines, start, header_end, language, symbol, "header")
            _walk_structure(children, lines, language, out, covered)
            continue
        _append_span(out, covered, lines, start, end, language, symbol, kind)


def _append_span(
    out: list[Chunk],
    covered: list[bool],
    lines: list[str],
    start: int,
    end: int,
    language: str | None,
    symbol: str | None,
    kind: str,
) -> None:
    for chunk in _emit_span(lines, start, end, language, symbol, kind):
        out.append(chunk)
        for line_no in range(chunk.start_line, chunk.end_line + 1):
            if 0 < line_no < len(covered):
                covered[line_no] = True


def _emit_span(
    lines: list[str],
    start: int,
    end: int,
    language: str | None,
    symbol: str | None,
    kind: str,
) -> list[Chunk]:
    start = max(1, start)
    end = min(len(lines), end)
    if end < start:
        return []
    if end - start + 1 <= STRUCTURAL_LINE_LIMIT:
        chunk = _make_chunk(lines, start, end, language, symbol, kind)
        return [chunk] if chunk.content.strip() else []
    chunks: list[Chunk] = []
    pos = start
    guard = 0
    while pos <= end and guard < 10000:
        guard += 1
        win_end = min(end, pos + WINDOW_LINES - 1)
        chunk = _make_chunk(lines, pos, win_end, language, symbol, f"{kind}_part")
        if chunk.content.strip():
            chunks.append(chunk)
        if win_end >= end:
            break
        next_pos = win_end - WINDOW_OVERLAP + 1
        pos = next_pos if next_pos > pos else pos + 1
    return chunks


def _fill_gaps(lines: list[str], covered: list[bool], language: str | None) -> list[Chunk]:
    chunks: list[Chunk] = []
    total = len(lines)
    index = 1
    while index <= total:
        if covered[index] or not lines[index - 1].strip():
            index += 1
            continue
        start = index
        while index <= total and not covered[index]:
            index += 1
        end = index - 1
        while end >= start and not lines[end - 1].strip():
            end -= 1
        if end < start:
            continue
        pos = start
        guard = 0
        while pos <= end and guard < 10000:
            guard += 1
            win_end = min(end, pos + WINDOW_LINES - 1)
            chunk = _make_chunk(lines, pos, win_end, language, None, "window")
            if chunk.content.strip():
                chunks.append(chunk)
            if win_end >= end:
                break
            next_pos = win_end - WINDOW_OVERLAP + 1
            pos = next_pos if next_pos > pos else pos + 1
    return chunks


def _make_chunk(
    lines: list[str],
    start: int,
    end: int,
    language: str | None,
    symbol: str | None,
    kind: str,
) -> Chunk:
    return Chunk(
        start_line=start,
        end_line=end,
        content="\n".join(lines[start - 1 : end]),
        language=language,
        symbol_name=symbol,
        chunk_kind=kind,
    )


def _attach_leading_decorators(lines: list[str], start: int) -> int:
    candidate = start
    saw_decorator = False
    index = start - 2
    while index >= 0:
        stripped = lines[index].strip()
        if _is_decorator_line(stripped):
            saw_decorator = True
            candidate = index + 1
            index -= 1
            continue
        if (
            stripped == ""
            and saw_decorator
            and index - 1 >= 0
            and _is_decorator_line(lines[index - 1].strip())
        ):
            index -= 1
            continue
        break
    return candidate if saw_decorator else start


def _is_decorator_line(stripped: str) -> bool:
    if stripped.startswith("@") or stripped.startswith("#["):
        return True
    return stripped.startswith("[") and stripped.endswith("]") and len(stripped) < 200


def _kind_label(kind: object) -> str:
    raw = getattr(kind, "name", None) or str(kind)
    label = str(raw).split(".")[-1].lower()
    if "method" in label:
        return "method"
    if label in {"function", "func"}:
        return "function"
    if label in {"class", "interface", "trait", "impl"}:
        return "class"
    if label in {"struct", "enum", "type", "typealias", "module"}:
        return "type"
    return "block"


def _clamp_line(line_no: int, total: int) -> int:
    return max(1, min(line_no, total))
