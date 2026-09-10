"""Two-layer citation evaluation: deterministic parsing and optional LLM semantics."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Settings
from .fetcher import WebFetcher
from .models import AtomicClaim, ClaimPatch, ClaimPatchBatch
from .providers import FakeLLMProvider, OpenAILLMProvider
from .search import MockSearchProvider
from .workflow import ResearchWorkflow

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "evals" / "citation_cases.jsonl"
DEFAULT_THRESHOLDS = ROOT / "evals" / "deterministic_thresholds.json"
DEFAULT_SEMANTIC_THRESHOLDS = ROOT / "evals" / "semantic_thresholds.json"
SEMANTIC_BASELINE_CASES = {
    "single-fetched",
}

def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def load_thresholds(path: Path) -> dict[str, float]:
    return {name: float(value) for name, value in json.loads(path.read_text()).items()}

def threshold_failures(metrics: dict, thresholds: dict[str, float]) -> dict[str, dict[str, float]]:
    return {name: {"actual": metrics.get(name, 0), "minimum": minimum} for name, minimum in thresholds.items() if metrics.get(name, 0) < minimum}

def _adjacent(text: str, source_ids: list[str]) -> bool:
    if not source_ids:
        return True
    labels = "".join(f"[{source_id}]" for source_id in source_ids)
    return bool(re.search(re.escape(labels) + r"\s*(?:[.!?。！？])?$", text))

async def evaluate_citations(path: Path = DEFAULT_CORPUS, live_model: bool = False, thresholds_path: Path = DEFAULT_THRESHOLDS, semantic_thresholds_path: Path = DEFAULT_SEMANTIC_THRESHOLDS) -> dict:
    cases = load_cases(path)
    settings = Settings.from_env() if live_model else Settings(llm_provider="mock", search_provider="mock")
    if live_model:
        if settings.llm_provider == "mock" or not settings.openai_api_key:
            raise ValueError("live-model evaluation requires a real LLM and OPENAI_API_KEY")
        llm = OpenAILLMProvider(settings.openai_api_key, settings.openai_base_url, settings.openai_model, settings)
    else:
        llm = FakeLLMProvider()
    workflow = ResearchWorkflow(llm, MockSearchProvider(), WebFetcher(), settings=settings)
    totals = {"cases": len(cases), "claims": 0, "structure": 0, "offsets": 0, "sources": 0, "quality": 0, "adjacency": 0, "integrity": 0, "patches": 0, "patch_ok": 0, "semantic_claims": 0, "verdict": 0}
    case_results = []
    semantic_jobs = []
    for case in cases:
        claims = workflow._atomic_claims(case["report"])
        expected = case["expected"]
        source_map = {source["source_id"]: source for source in case["sources"]}
        totals["claims"] += len(expected)
        exact = len(claims) == len(expected) and all(claim.text == item["text"] for claim, item in zip(claims, expected, strict=True))
        totals["structure"] += exact
        for claim, item in zip(claims, expected, strict=False):
            totals["offsets"] += case["report"][claim.start_offset:claim.end_offset] == claim.text
            totals["sources"] += claim.source_ids == item["source_ids"]
            totals["quality"] += workflow._evidence_quality(claim.source_ids, source_map) == item["evidence_quality"]
            totals["adjacency"] += _adjacent(claim.text, item["source_ids"])
            totals["integrity"] += all(source_id in source_map for source_id in claim.source_ids) == item.get("citations_exist", True)
        if patch := case.get("patch"):
            totals["patches"] += 1
            problem = {claim.claim_id: claim for claim in claims if claim.claim_id == patch["claim_id"]}
            revised, applied = workflow._apply_claim_patches(case["report"], ClaimPatchBatch(patches=[ClaimPatch(**patch["operation"])]), problem, source_map)
            totals["patch_ok"] += (
                applied == 1
                and patch["unchanged_text"] in revised
                and revised == patch["expected_report"]
            )
        result = {"case_id": case["case_id"], "claims": len(claims), "structure_correct": exact}
        if live_model and case["case_id"] in SEMANTIC_BASELINE_CASES:
            semantic_jobs.append((claims, source_map, expected, result))
            totals["semantic_claims"] += len(expected)
        case_results.append(result)
    if semantic_jobs:
        batch_claims: list[AtomicClaim] = []
        batch_sources = {}
        expected_verdicts = []
        result_ranges = []
        for case_index, (claims, source_map, expected, result) in enumerate(semantic_jobs, 1):
            range_start = len(batch_claims)
            renamed = {source_id: f"C{case_index}_{source_id}" for source_id in source_map}
            for source_id, source in source_map.items():
                batch_sources[renamed[source_id]] = {**source, "source_id": renamed[source_id]}
            for claim, item in zip(claims, expected, strict=False):
                text = claim.text
                source_ids = []
                for source_id in claim.source_ids:
                    namespaced = renamed.get(source_id, f"C{case_index}_{source_id}")
                    text = text.replace(f"[{source_id}]", f"[{namespaced}]")
                    source_ids.append(namespaced)
                batch_claims.append(claim.model_copy(update={
                    "claim_id": f"C{case_index}_{claim.claim_id}",
                    "text": text,
                    "source_ids": source_ids,
                }))
                expected_verdicts.append(item["verdict"])
            result_ranges.append((range_start, len(batch_claims), result))
        audit = await workflow._batch_citation_audit(batch_claims, batch_sources)
        verdict_matches = [
            check.verdict == expected
            for check, expected in zip(audit.checks, expected_verdicts, strict=True)
        ]
        totals["verdict"] = sum(verdict_matches)
        for start, end, result in result_ranges:
            result["semantic_verdict_correct"] = sum(verdict_matches[start:end])
    count = totals["claims"]
    deterministic = {
        "cases": totals["cases"], "claims": count,
        "atomic_claim_exact_match": totals["structure"] / totals["cases"] if totals["cases"] else 0,
        "source_offset_accuracy": totals["offsets"] / count if count else 0,
        "source_set_exact_match": totals["sources"] / count if count else 0,
        "evidence_quality_accuracy": totals["quality"] / count if count else 0,
        "citation_adjacency_accuracy": totals["adjacency"] / count if count else 0,
        "citation_integrity_accuracy": totals["integrity"] / count if count else 0,
        "patch_preservation_accuracy": totals["patch_ok"] / totals["patches"] if totals["patches"] else 1.0,
    }
    semantic_count = totals["semantic_claims"]
    semantic = {
        "semantic_cases": len(semantic_jobs),
        "semantic_claims": semantic_count,
        "semantic_verdict_accuracy": totals["verdict"] / semantic_count if live_model and semantic_count else None,
        "live_model": live_model,
        "model_calls": llm.calls if live_model else 0,
    }
    deterministic_thresholds = load_thresholds(thresholds_path)
    semantic_thresholds = load_thresholds(semantic_thresholds_path)
    failures = threshold_failures(deterministic, deterministic_thresholds)
    semantic_failures = threshold_failures(semantic, semantic_thresholds) if live_model else {}
    return {"metrics": {**deterministic, **semantic}, "metric_layers": {"deterministic": deterministic, "semantic": semantic}, "thresholds": {"deterministic": deterministic_thresholds, "semantic": semantic_thresholds}, "threshold_failures": {**failures, **semantic_failures}, "cases": case_results}
