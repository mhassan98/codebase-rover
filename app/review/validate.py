from __future__ import annotations

from dataclasses import dataclass, field

from app.review.diffparse import DiffFile

_SEVERITY = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "blocker": "critical",
    "info": "low",
}
_SPECULATIVE = {"speculative", "possible", "uncertain", "guess", "hypothesis", "maybe"}
_ACCEPTED_CONFIDENCE = {"confirmed", "likely"}


@dataclass
class Finding:
    severity: str
    confidence: str
    title: str
    explanation: str
    file_path: str
    start_line: int
    end_line: int
    evidence: list[str]


@dataclass
class ReviewAssessment:
    summary: str
    findings: list[Finding] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    synthesis: str = "model"


def validate_model_review(
    payload: dict,
    diff_files: list[DiffFile],
    diff_text: str,
    head_texts: dict[str, str],
) -> ReviewAssessment:
    summary = payload.get("summary") if isinstance(payload.get("summary"), str) else ""
    assumptions = _string_list(payload.get("assumptions"))
    limitations = _string_list(payload.get("limitations"))
    by_path = _index_files(diff_files)
    findings: list[Finding] = []
    rejected: list[dict] = []
    raw_findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    for raw in raw_findings[:40]:
        if len(findings) >= 30:
            break
        if not isinstance(raw, dict):
            rejected.append({"title": "", "reason": "finding was not an object"})
            continue
        title = _short(raw.get("title"))
        decision = _classify_finding(raw, title, by_path, diff_text, head_texts)
        if isinstance(decision, Finding):
            findings.append(decision)
        elif decision[0] == "assumption":
            assumptions.append(decision[1])
        else:
            rejected.append({"title": title, "reason": decision[1]})
    return ReviewAssessment(
        summary=summary.strip()[:2000],
        findings=findings,
        assumptions=assumptions[:20],
        limitations=limitations[:20],
        rejected=rejected[:20],
    )


def _classify_finding(raw: dict, title: str, by_path: dict[str, DiffFile], diff_text: str, head_texts: dict[str, str]):
    explanation = raw.get("explanation") if isinstance(raw.get("explanation"), str) else ""
    if not title or not explanation.strip():
        return ("rejected", "title and explanation are required")
    severity = _SEVERITY.get(str(raw.get("severity", "")).lower())
    if severity is None:
        return ("rejected", "severity is missing or unknown")
    confidence = str(raw.get("confidence", "")).strip().lower()
    evidence = [item.strip() for item in raw.get("evidence", []) if isinstance(item, str) and item.strip()] if isinstance(raw.get("evidence"), list) else []
    file_path = raw.get("file_path") if isinstance(raw.get("file_path"), str) else ""
    diff_file = _resolve_file(file_path, by_path)
    if diff_file is None:
        return ("rejected", "file is not part of the diff")
    try:
        start_line = int(raw.get("start_line"))
        end_line = int(raw.get("end_line"))
    except (TypeError, ValueError):
        return ("rejected", "line numbers are missing")
    if end_line < start_line:
        return ("rejected", "line range is reversed")
    if not _anchors(diff_file, start_line, end_line):
        return ("rejected", "lines are outside the changed hunks")
    haystacks = [diff_text, head_texts.get(diff_file.path, ""), head_texts.get(file_path, "")]
    evidence_ok = _evidence_found(evidence, haystacks)
    if confidence in _SPECULATIVE or not confidence:
        if confidence in _SPECULATIVE:
            return ("assumption", f"Not reported as a defect: {title}. {explanation.strip()}")
        confidence = "likely"
    if confidence not in _ACCEPTED_CONFIDENCE:
        return ("assumption", f"Not reported as a defect: {title}. {explanation.strip()}")
    if not evidence_ok:
        return (
            "assumption",
            f"Not reported as a defect: {title}. The evidence quote was not found in the diff or head revision.",
        )
    if confidence == "confirmed" and not evidence_ok:
        return ("assumption", f"Not reported as a defect: {title}.")
    return Finding(
        severity=severity,
        confidence=confidence,
        title=title,
        explanation=explanation.strip()[:4000],
        file_path=diff_file.path,
        start_line=start_line,
        end_line=end_line,
        evidence=evidence[:6],
    )


def _anchors(diff_file: DiffFile, start_line: int, end_line: int) -> bool:
    for hunk in diff_file.hunks:
        numbers = [
            number
            for line in hunk.lines
            for number in (line.old_line, line.new_line)
            if number is not None
        ]
        if not numbers:
            continue
        if start_line <= max(numbers) and end_line >= min(numbers):
            return True
    return False


def _evidence_found(quotes: list[str], haystacks: list[str]) -> bool:
    normalized = [_normalize(value) for value in haystacks if value]
    for quote in quotes:
        needle = _normalize(quote)
        if len(needle) < 12:
            continue
        probe = needle[:180]
        if any(probe in haystack for haystack in normalized):
            return True
    return False


def _normalize(value: str) -> str:
    return " ".join(value.replace("“", '"').replace("”", '"').replace("’", "'").split())


def _index_files(files: list[DiffFile]) -> dict[str, DiffFile]:
    indexed: dict[str, DiffFile] = {}
    for item in files:
        if item.new_path:
            indexed[item.new_path] = item
        if item.old_path and item.old_path not in indexed:
            indexed[item.old_path] = item
    return indexed


def _resolve_file(path: str, by_path: dict[str, DiffFile]) -> DiffFile | None:
    candidate = path.strip().replace("\\", "/")
    if candidate.startswith("./"):
        candidate = candidate[2:]
    if candidate.startswith("a/") or candidate.startswith("b/"):
        alt = candidate[2:]
        return by_path.get(candidate) or by_path.get(alt)
    return by_path.get(candidate)


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip()[:1000] for item in value if isinstance(item, str) and item.strip()]


def _short(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:200]
