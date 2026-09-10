"""Deterministic offline provider used by tests and demos."""

import json
import re
from pathlib import Path

from ..models import (
    CitationCheck,
    Evidence,
    EvidenceEvaluation,
    RefinedQueries,
    ResearchPlan,
    Source,
    SubQuestion,
    SubquestionAssessment,
)
from .base import LLMProvider, T


class FakeLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.model = "fake"
        self.calls = self.input_tokens = self.output_tokens = self.total_tokens = 0
        self.reasoning_tokens = self.cached_tokens = self.retries = 0
        self.call_metrics = []

    async def structured(self, prompt: str, schema: type[T], *, node: str = "structured") -> T:
        self.calls += 1
        self.input_tokens += len(prompt.split())
        self.total_tokens = self.input_tokens + self.output_tokens
        self.call_metrics.append({"node": node, "model": self.model, "finish_reason": "stop"})
        if schema is ResearchPlan:
            question = prompt.split("QUESTION:", 1)[-1].splitlines()[0].strip()
            return ResearchPlan(
                objective=f"Build a verifiable answer to: {question}",
                subquestions=[
                    SubQuestion(question=f"What are the core concepts behind {question}?", search_queries=[f"{question} core concepts"]),
                    SubQuestion(question=f"What evidence supports {question}?", search_queries=[f"{question} evidence sources"]),
                    SubQuestion(question=f"What limitations and risks affect {question}?", search_queries=[f"{question} limitations risks"]),
                ],
            )  # type: ignore[return-value]
        if schema is RefinedQueries:
            return RefinedQueries(queries=["independent evidence agent reliability"])  # type: ignore[return-value]
        if schema is EvidenceEvaluation:
            questions = re.findall(r"^SUBQUESTION: (.+)$", prompt, re.MULTILINE)
            return EvidenceEvaluation(assessments=[
                SubquestionAssessment(subquestion=q, sufficient=True, reason="Relevant bundled sources were found.", source_count=2)
                for q in questions
            ], sufficient=True)  # type: ignore[return-value]
        if schema is CitationCheck:
            return CitationCheck(claim_id="CL1", source_ids=["S1"], claim="claim", verdict="supported", evidence_quality="fetched", reason="The source contains matching evidence.")  # type: ignore[return-value]
        raise TypeError(f"FakeLLMProvider has no response for {schema.__name__}")

    async def text(self, prompt: str, *, node: str = "text") -> str:
        self.calls += 1
        self.input_tokens += len(prompt.split())
        self.total_tokens = self.input_tokens + self.output_tokens
        self.call_metrics.append({"node": node, "model": self.model, "finish_reason": "stop"})
        return ""

    def extract_evidence(self, sources: list[Source], subquestions: list[str]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for index, question in enumerate(subquestions):
            for source in sources[index % len(sources):index % len(sources) + 2]:
                sentence = source.content.split(". ")[0].strip() + "."
                evidence.append(Evidence(claim=sentence, evidence=sentence, source_id=source.source_id, relevance_score=0.85, subquestion=question))
        return evidence


def load_mock_documents() -> list[dict[str, str]]:
    path = Path(__file__).resolve().parents[3] / "examples" / "mock_sources.json"
    return json.loads(path.read_text(encoding="utf-8"))

__all__ = ["FakeLLMProvider", "load_mock_documents"]
