from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubQuestion(StrictModel):
    question: str = Field(min_length=5)
    search_queries: list[str] = Field(min_length=1, max_length=3)

    @field_validator("search_queries")
    @classmethod
    def unique_queries(cls, values: list[str]) -> list[str]:
        clean = [v.strip() for v in values if v.strip()]
        if len(clean) != len({v.casefold() for v in clean}):
            raise ValueError("search queries must be unique")
        return clean


class ResearchPlan(StrictModel):
    objective: str = Field(min_length=5)
    subquestions: list[SubQuestion] = Field(min_length=3, max_length=5)


class SearchResult(StrictModel):
    title: str
    url: HttpUrl
    snippet: str = ""
    query: str = ""


class Source(StrictModel):
    source_id: str
    title: str
    url: HttpUrl
    content: str
    query: str = ""
    source_status: Literal["search_result", "fetched", "snippet_fallback"] = "fetched"
    final_url: HttpUrl | None = None
    http_status: int | None = None
    content_type: str = ""
    retrieved_at: str = ""
    content_hash: str = ""
    parser: str = ""
    quality_score: float = Field(default=0.5, ge=0, le=1)


class Evidence(StrictModel):
    claim: str
    evidence: str
    source_id: str
    relevance_score: float = Field(ge=0, le=1)
    subquestion: str


class EvidenceBatch(StrictModel):
    evidence: list[Evidence] = Field(default_factory=list)


class SubquestionAssessment(StrictModel):
    subquestion: str
    sufficient: bool
    reason: str
    source_count: int = Field(ge=0)


class EvidenceEvaluation(StrictModel):
    assessments: list[SubquestionAssessment]
    missing_information: list[str] = Field(default_factory=list)
    sufficient: bool


class AtomicClaim(StrictModel):
    claim_id: str
    text: str
    source_ids: list[str] = Field(min_length=1)
    node_type: str = "paragraph"
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    start_line: int = Field(default=0, ge=0)
    end_line: int = Field(default=0, ge=0)


class CitationCheck(StrictModel):
    claim_id: str
    source_ids: list[str] = Field(min_length=1)
    claim: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    evidence_quality: Literal["fetched", "snippet_fallback", "mixed", "unavailable"]
    reason: str


class CitationDecision(StrictModel):
    status: Literal["supported", "partially_supported", "unsupported"]
    reason: str


class CitationBatchItem(StrictModel):
    claim_id: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    reason: str


class CitationBatch(StrictModel):
    checks: list[CitationBatchItem] = Field(default_factory=list)


class ClaimPatch(StrictModel):
    claim_id: str
    action: Literal["replace", "delete"]
    replacement: str = ""


class ClaimPatchBatch(StrictModel):
    patches: list[ClaimPatch] = Field(default_factory=list)


class CitationAudit(StrictModel):
    checks: list[CitationCheck] = Field(default_factory=list)
    valid_citations: int = 0
    total_citations: int = 0
    coverage: float = Field(default=0, ge=0, le=1)


class Metrics(StrictModel):
    started_at: float
    elapsed_seconds: float = 0
    search_count: int = 0
    fetch_count: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    retry_count: int = 0
    model: str | None = None
    node_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    call_metrics: list[dict[str, Any]] = Field(default_factory=list)


class ResearchState(TypedDict, total=False):
    task_id: str
    question: str
    research_plan: dict[str, Any]
    queries: list[str]
    completed_queries: list[str]
    search_results: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    missing_information: list[str]
    research_round: int
    max_rounds: int
    report: str
    citation_audit: dict[str, Any]
    citation_revision: dict[str, Any]
    errors: list[str]
    metrics: dict[str, Any]
    evaluation: dict[str, Any]
    resume_from: str


class RefinedQueries(StrictModel):
    queries: list[str] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def non_empty(self) -> RefinedQueries:
        self.queries = list(dict.fromkeys(q.strip() for q in self.queries if q.strip()))
        if not self.queries:
            raise ValueError("at least one non-empty query is required")
        return self
