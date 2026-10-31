from app.config import Settings
from app.qa.service import INSUFFICIENT_EVIDENCE, synthesize_answer
from app.retrieval.source import VerifiedExcerpt
from app.review.diffparse import parse_unified_diff
from app.review.service import synthesize_review
from tests.test_diffparse import DIFF


class FakeChat:
    provider_name = "fake"
    model_name = "fake-1"

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0
        self.messages = None

    def complete(self, messages, *, context_stats):
        self.calls += 1
        self.messages = messages
        import json

        return json.dumps(self.payload)


def _settings() -> Settings:
    return Settings(max_context_chunks=8, max_retrieved_chars=8000, max_diff_prompt_chars=8000, max_prompt_chars=20000)


def _excerpt() -> VerifiedExcerpt:
    return VerifiedExcerpt(
        citation_id="c1",
        repository_id="repo",
        repository_name="fastcrib",
        file_path="src/tax.py",
        start_line=4,
        end_line=6,
        commit_sha="abc",
        text="def tax(amount):\n    return amount * 0.08",
        index_drift=False,
    )


def test_answer_without_excerpts_does_not_call_model() -> None:
    provider = FakeChat({"sufficient": True, "answer": "invented", "citation_ids": ["c1"]})
    draft = synthesize_answer("How does tax work?", [], provider, _settings())
    assert provider.calls == 0
    assert draft.sufficient is False
    assert draft.answer == INSUFFICIENT_EVIDENCE


def test_answer_rejects_unknown_citations() -> None:
    provider = FakeChat({"sufficient": True, "answer": "Tax multiplies by 0.08.", "citation_ids": ["missing"]})
    draft = synthesize_answer("How does tax work?", [_excerpt()], provider, _settings())
    assert draft.sufficient is False
    assert draft.citations == []


def test_answer_keeps_supported_citation() -> None:
    provider = FakeChat({"sufficient": True, "answer": "Tax multiplies the amount by 0.08.", "citation_ids": ["c1"]})
    draft = synthesize_answer("How does tax work?", [_excerpt()], provider, _settings())
    assert draft.sufficient is True
    assert draft.citations[0].commit_sha == "abc"
    assert "UNRELATED_SECRET" not in provider.messages[1]["content"]
    assert "return amount * 0.08" in provider.messages[1]["content"]


def test_review_keeps_only_anchored_findings() -> None:
    parsed = parse_unified_diff(DIFF)
    added = next(line for item in parsed for hunk in item.hunks for line in hunk.lines if "negative amount" in line.text)
    provider = FakeChat(
        {
            "summary": "Negative amounts are rejected.",
            "findings": [
                {
                    "severity": "high",
                    "confidence": "confirmed",
                    "title": "Negative amounts raise",
                    "explanation": "The diff raises ValueError for a negative amount.",
                    "file_path": "billing/tax.py",
                    "start_line": added.new_line,
                    "end_line": added.new_line,
                    "evidence": ['raise ValueError("negative amount")'],
                },
                {
                    "severity": "low",
                    "confidence": "speculative",
                    "title": "Maybe the rate is illegal",
                    "explanation": "No source says the rate is illegal.",
                    "file_path": "billing/tax.py",
                    "start_line": added.new_line,
                    "end_line": added.new_line,
                    "evidence": ["0.08"],
                },
            ],
            "assumptions": [],
            "limitations": [],
        }
    )
    assessment = synthesize_review(DIFF, parsed, {"billing/tax.py": "raise ValueError(\"negative amount\")"}, [], provider, _settings())
    assert [item.title for item in assessment.findings] == ["Negative amounts raise"]
    assert any("Maybe the rate is illegal" in item for item in assessment.assumptions)
    assert "UNRELATED_SECRET" not in provider.messages[1]["content"]
    assert "negative amount" in provider.messages[1]["content"]
