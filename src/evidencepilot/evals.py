from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .fetcher import WebFetcher
from .providers import FakeLLMProvider, OpenAILLMProvider
from .search import MockSearchProvider
from .workflow import ResearchWorkflow


async def evaluate_citations(path: Path, live_model: bool = False) -> dict:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    settings = Settings.from_env() if live_model else Settings(
        llm_provider="mock", search_provider="mock"
    )
    if live_model:
        if settings.llm_provider == "mock" or not settings.openai_api_key:
            raise ValueError("live-model evaluation requires a real LLM and OPENAI_API_KEY")
        llm = OpenAILLMProvider(
            settings.openai_api_key, settings.openai_base_url, settings.openai_model, settings
        )
    else:
        llm = FakeLLMProvider()
    workflow = ResearchWorkflow(llm, MockSearchProvider(), WebFetcher(), settings=settings)
    totals = {"claims": 0, "structure": 0, "sources": 0, "quality": 0, "verdict": 0}
    results = []
    for case in cases:
        claims = workflow._atomic_claims(case["report"])
        expected = case["expected"]
        totals["claims"] += len(expected)
        source_map = {source["source_id"]: source for source in case["sources"]}
        for index, item in enumerate(expected):
            if index >= len(claims):
                continue
            claim = claims[index]
            totals["structure"] += claim.text == item["text"]
            totals["sources"] += claim.source_ids == item["source_ids"]
            totals["quality"] += (
                workflow._evidence_quality(claim.source_ids, source_map)
                == item["evidence_quality"]
            )
        result = {"case_id": case["case_id"], "claims": len(claims)}
        if live_model:
            audit = await workflow._batch_citation_audit(claims, source_map)
            actual = {check.claim: check.verdict for check in audit.checks}
            matched = sum(actual.get(item["text"]) == item["verdict"] for item in expected)
            totals["verdict"] += matched
            result["semantic_verdict_correct"] = matched
        results.append(result)
    count = totals["claims"]
    return {
        "metrics": {
            "cases": len(cases), "claims": count,
            "atomic_claim_exact_match": totals["structure"] / count if count else 0,
            "source_set_exact_match": totals["sources"] / count if count else 0,
            "evidence_quality_accuracy": totals["quality"] / count if count else 0,
            "semantic_verdict_accuracy": totals["verdict"] / count if live_model and count else None,
            "live_model": live_model, "model_calls": llm.calls if live_model else 0,
        },
        "cases": results,
    }
