from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    llm_provider: str = "openai"
    search_provider: str = "tavily"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    tavily_api_key: str | None = None
    database_path: str = "data/evidencepilot.db"
    max_queries_per_round: int = 8
    results_per_query: int = 4
    max_sources: int = 20
    max_page_bytes: int = 2_000_000
    max_source_chars: int = 20_000
    http_timeout_seconds: float = 12.0
    llm_thinking_mode: str = "disabled"
    llm_reasoning_effort: str = "low"
    llm_max_tokens_structured: int = 2048
    llm_max_tokens_citation_audit: int = 4096
    llm_max_tokens_report: int = 4096
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2

    def __post_init__(self) -> None:
        if self.llm_provider not in {"openai", "deepseek", "mock"}:
            raise ValueError("LLM_PROVIDER must be 'openai', 'deepseek', or 'mock'")
        if self.search_provider not in {"tavily", "mock"}:
            raise ValueError("SEARCH_PROVIDER must be 'tavily' or 'mock'")
        if self.llm_thinking_mode not in {"disabled", "enabled"}:
            raise ValueError("LLM_THINKING_MODE must be 'disabled' or 'enabled'")
        if min(self.llm_max_tokens_structured, self.llm_max_tokens_citation_audit, self.llm_max_tokens_report) < 1:
            raise ValueError("LLM token budgets must be positive")
        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative")

    def token_budget(self, node: str) -> int:
        """Central node budget policy; values are caps, not usage targets."""
        overrides = {
            "evaluate_evidence": min(1536, self.llm_max_tokens_structured),
            "refine_queries": min(1024, self.llm_max_tokens_structured),
            "verify_citations": self.llm_max_tokens_citation_audit,
            "write_report": self.llm_max_tokens_report,
            "revise_unsupported_claims": self.llm_max_tokens_report,
        }
        return overrides.get(node, self.llm_max_tokens_structured)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "").casefold(),
            search_provider=os.getenv("SEARCH_PROVIDER", "").casefold(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            database_path=os.getenv("EVIDENCEPILOT_DB", "data/evidencepilot.db"),
            max_queries_per_round=int(os.getenv("MAX_QUERIES_PER_ROUND", "8")),
            results_per_query=int(os.getenv("RESULTS_PER_QUERY", "4")),
            max_sources=int(os.getenv("MAX_SOURCES", "20")),
            llm_thinking_mode=os.getenv("LLM_THINKING_MODE", "disabled").casefold(),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
            llm_max_tokens_structured=int(os.getenv("LLM_MAX_TOKENS_STRUCTURED", "2048")),
            llm_max_tokens_citation_audit=int(os.getenv("LLM_MAX_TOKENS_CITATION_AUDIT", "4096")),
            llm_max_tokens_report=int(os.getenv("LLM_MAX_TOKENS_REPORT", "4096")),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )
