from __future__ import annotations

import re

_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-~+/]+=*")
_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|password|secret)\b\s*[=:]\s*\S+"
)


def redact(text: str) -> str:
    cleaned = _BEARER.sub(r"\1[redacted]", text)
    cleaned = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[redacted]", cleaned)
    return cleaned
