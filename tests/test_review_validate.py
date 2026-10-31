from app.review.diffparse import parse_unified_diff
from app.review.validate import validate_model_review
from tests.test_diffparse import DIFF


def test_speculative_and_unquoted_findings_are_not_defects() -> None:
    parsed = parse_unified_diff(DIFF)[0]
    added = next(line for hunk in parsed.hunks for line in hunk.lines if "negative amount" in line.text)
    payload = {
        "summary": "Tax now rejects negative amounts.",
        "findings": [
            {
                "severity": "high",
                "confidence": "confirmed",
                "title": "Negative amounts raise",
                "explanation": "The new branch rejects negative amounts.",
                "file_path": "billing/tax.py",
                "start_line": added.new_line,
                "end_line": added.new_line,
                "evidence": ['raise ValueError("negative amount")'],
            },
            {
                "severity": "low",
                "confidence": "speculative",
                "title": "Currency might be wrong",
                "explanation": "Nothing in the diff shows a currency defect.",
                "file_path": "billing/tax.py",
                "start_line": added.new_line,
                "end_line": added.new_line,
                "evidence": ["USD"],
            },
            {
                "severity": "high",
                "confidence": "confirmed",
                "title": "Invented crash",
                "explanation": "This quote is not in the change.",
                "file_path": "billing/tax.py",
                "start_line": added.new_line,
                "end_line": added.new_line,
                "evidence": ["this quote is not present in the diff"],
            },
            {
                "severity": "medium",
                "confidence": "confirmed",
                "title": "Wrong file",
                "explanation": "Not in the diff.",
                "file_path": "other.py",
                "start_line": 1,
                "end_line": 1,
                "evidence": ['raise ValueError("negative amount")'],
            },
        ],
        "assumptions": [],
        "limitations": [],
    }
    result = validate_model_review(payload, [parsed], DIFF, {"billing/tax.py": DIFF})
    assert [item.title for item in result.findings] == ["Negative amounts raise"]
    assert result.findings[0].confidence == "confirmed"
    assert result.findings[0].file_path == "billing/tax.py"
    assert any("Currency might be wrong" in item for item in result.assumptions)
    assert any("Invented crash" in item for item in result.assumptions)
    assert any(item["title"] == "Wrong file" for item in result.rejected)
