from pathlib import Path

import pytest

from evidencepilot.evals import evaluate_citations


@pytest.mark.asyncio
async def test_cli_and_script_share_fixed_corpus_evaluation():
    root = Path(__file__).resolve().parents[1]
    result = await evaluate_citations(root / "evals" / "citation_cases.jsonl")
    assert result["metrics"]["cases"] == 32
    assert result["metrics"]["claims"] == 46
    assert result["metrics"]["citation_adjacency_accuracy"] == 1
    assert result["metrics"]["citation_integrity_accuracy"] == 1
    assert result["threshold_failures"] == {}
    assert result["metrics"]["patch_preservation_accuracy"] == 1
    assert result["metric_layers"]["semantic"]["semantic_verdict_accuracy"] is None
    assert result["thresholds"]["deterministic"]["atomic_claim_exact_match"] == 1
    assert result["thresholds"]["semantic"] == {}
