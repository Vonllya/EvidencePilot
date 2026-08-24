"""Reproducible citation evaluation; real-model scoring is explicit and opt-in."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from evidencepilot.config import Settings
from evidencepilot.fetcher import WebFetcher
from evidencepilot.providers import FakeLLMProvider, OpenAILLMProvider
from evidencepilot.search import MockSearchProvider
from evidencepilot.workflow import ResearchWorkflow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "evals" / "citation_cases.jsonl"


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def evaluate(path: Path, live_model: bool) -> dict:
    cases = load_cases(path)
    if live_model:
        settings = Settings.from_env()
        if settings.llm_provider == "mock" or not settings.openai_api_key:
            raise SystemExit("--live-model requires a real LLM_PROVIDER and OPENAI_API_KEY")
        llm = OpenAILLMProvider(
            settings.openai_api_key, settings.openai_base_url, settings.openai_model, settings
        )
    else:
        settings = Settings(llm_provider="mock", search_provider="mock")
        llm = FakeLLMProvider()
    workflow = ResearchWorkflow(llm, MockSearchProvider(), WebFetcher(), settings=settings)
    claim_total = structure_correct = source_set_correct = quality_correct = 0
    verdict_total = verdict_correct = 0
    case_results = []
    for case in cases:
        claims = workflow._atomic_claims(case["report"])
        expected = case["expected"]
        claim_total += len(expected)
        structure_correct += sum(
            index < len(claims) and claims[index].text == item["text"]
            for index, item in enumerate(expected)
        )
        source_set_correct += sum(
            index < len(claims) and claims[index].source_ids == item["source_ids"]
            for index, item in enumerate(expected)
        )
        source_map = {source["source_id"]: source for source in case["sources"]}
        quality_correct += sum(
            index < len(claims)
            and workflow._evidence_quality(claims[index].source_ids, source_map)
            == item["evidence_quality"]
            for index, item in enumerate(expected)
        )
        result = {"case_id": case["case_id"], "claims": len(claims)}
        if live_model:
            audit = await workflow._batch_citation_audit(claims, source_map)
            actual = {check.claim: check.verdict for check in audit.checks}
            verdict_total += len(expected)
            matched = sum(actual.get(item["text"]) == item["verdict"] for item in expected)
            verdict_correct += matched
            result["semantic_verdict_correct"] = matched
        case_results.append(result)
    metrics = {
        "cases": len(cases),
        "claims": claim_total,
        "atomic_claim_exact_match": structure_correct / claim_total if claim_total else 0,
        "source_set_exact_match": source_set_correct / claim_total if claim_total else 0,
        "evidence_quality_accuracy": quality_correct / claim_total if claim_total else 0,
        "semantic_verdict_accuracy": (
            verdict_correct / verdict_total if live_model and verdict_total else None
        ),
        "live_model": live_model,
        "model_calls": llm.calls if live_model else 0,
    }
    return {"metrics": metrics, "cases": case_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--live-model", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    result = asyncio.run(evaluate(args.corpus, args.live_model))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    deterministic = result["metrics"]
    if any(deterministic[name] != 1 for name in (
        "atomic_claim_exact_match", "source_set_exact_match", "evidence_quality_accuracy"
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
