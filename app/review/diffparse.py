from __future__ import annotations

import re
from dataclasses import dataclass, field


class DiffParseError(ValueError):
    pass


@dataclass
class DiffLine:
    kind: str
    text: str
    old_line: int | None
    new_line: int | None


@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class DiffFile:
    old_path: str | None
    new_path: str | None
    status: str
    hunks: list[DiffHunk] = field(default_factory=list)
    binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(diff: str) -> list[DiffFile]:
    if diff is None:
        raise DiffParseError("diff is missing")
    if len(diff) > 2_000_000:
        raise DiffParseError("diff is too large to parse")
    if not diff.strip():
        return []
    lines = diff.splitlines()
    files: list[DiffFile] = []
    current: DiffFile | None = None
    hunk: DiffHunk | None = None
    old_cursor = 0
    new_cursor = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if line.startswith("diff --git "):
            old_path, new_path = _split_git_header(line)
            current = DiffFile(old_path=old_path, new_path=new_path, status="modified")
            files.append(current)
            hunk = None
            index += 1
            continue
        if _is_new_file_header(line, next_line) and (current is None or hunk is not None or current.hunks):
            current = DiffFile(old_path=None, new_path=None, status="modified")
            files.append(current)
            hunk = None
        if current is None:
            index += 1
            continue
        if line.startswith("new file mode"):
            current.status = "added"
        elif line.startswith("deleted file mode"):
            current.status = "deleted"
        elif line.startswith("rename from "):
            current.old_path = line[len("rename from ") :]
            current.status = "renamed"
        elif line.startswith("rename to "):
            current.new_path = line[len("rename to ") :]
            current.status = "renamed"
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            current.binary = True
            hunk = None
        elif line.startswith("--- ") and hunk is None:
            path = _parse_path(line[4:])
            if path == "/dev/null":
                current.status = "added"
                current.old_path = None
            else:
                current.old_path = path
        elif line.startswith("+++ ") and hunk is None:
            path = _parse_path(line[4:])
            if path == "/dev/null":
                current.status = "deleted"
                current.new_path = None
            else:
                current.new_path = path
        else:
            match = _HUNK.match(line)
            if match:
                old_count = int(match.group(2) if match.group(2) is not None else "1")
                new_count = int(match.group(4) if match.group(4) is not None else "1")
                hunk = DiffHunk(
                    old_start=int(match.group(1)),
                    old_count=old_count,
                    new_start=int(match.group(3)),
                    new_count=new_count,
                )
                current.hunks.append(hunk)
                old_cursor = hunk.old_start
                new_cursor = hunk.new_start
            elif hunk is not None and line.startswith("\\"):
                pass
            elif hunk is not None and line.startswith("+"):
                hunk.lines.append(DiffLine("add", line[1:], None, new_cursor))
                new_cursor += 1
            elif hunk is not None and line.startswith("-"):
                hunk.lines.append(DiffLine("del", line[1:], old_cursor, None))
                old_cursor += 1
            elif hunk is not None and line.startswith(" "):
                hunk.lines.append(DiffLine("context", line[1:], old_cursor, new_cursor))
                old_cursor += 1
                new_cursor += 1
        index += 1
    if not files:
        raise DiffParseError("the submission is not a unified diff")
    for item in files:
        if item.status == "modified" and item.old_path is None and item.new_path:
            item.status = "added"
        elif item.status == "modified" and item.new_path is None and item.old_path:
            item.status = "deleted"
    return files


def summarize_files(files: list[DiffFile]) -> list[dict]:
    summaries = []
    for item in files:
        added = sum(1 for hunk in item.hunks for line in hunk.lines if line.kind == "add")
        deleted = sum(1 for hunk in item.hunks for line in hunk.lines if line.kind == "del")
        summaries.append(
            {
                "path": item.path,
                "old_path": item.old_path,
                "new_path": item.new_path,
                "status": item.status,
                "binary": item.binary,
                "added_lines": added,
                "deleted_lines": deleted,
            }
        )
    return summaries


def _is_new_file_header(line: str, next_line: str) -> bool:
    return line.startswith("--- ") and next_line.startswith("+++ ")


def _parse_path(raw: str) -> str:
    token = raw.split("\t", 1)[0].strip()
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        token = token[1:-1]
    if token == "/dev/null":
        return token
    if token.startswith("a/") or token.startswith("b/"):
        return token[2:]
    return token


def _split_git_header(line: str) -> tuple[str | None, str | None]:
    rest = line[len("diff --git ") :]
    if rest.startswith('"'):
        try:
            old, remainder = _read_quoted(rest)
            new, _unused = _read_quoted(remainder.strip())
        except ValueError:
            return None, None
        return _strip_ab(old), _strip_ab(new)
    if rest.startswith("a/") and " b/" in rest:
        old, new = rest.split(" b/", 1)
        return old[2:], new
    return None, None


def _strip_ab(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _read_quoted(value: str) -> tuple[str, str]:
    if not value.startswith('"'):
        raise ValueError("expected quote")
    chars: list[str] = []
    index = 1
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            chars.append(value[index + 1])
            index += 2
            continue
        if char == '"':
            return "".join(chars), value[index + 1 :]
        chars.append(char)
        index += 1
    raise ValueError("unterminated quote")
