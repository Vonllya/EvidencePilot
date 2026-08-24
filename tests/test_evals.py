import json
from pathlib import Path

import pytest

from evidencepilot.evals import evaluate_citations


@pytest.mark.asyncio
async def test_cli_and_script_share_fixed_corpus_evaluation():
    root = Path(__file__).resolve().parents[1]
    result = await evaluate_citations(root / "evals" / "citation_cases.jsonl")
    assert result["metrics"]["claims"] == 7
    assert result["metrics"]["citation_adjacency_accuracy"] == 1
    assert result["metrics"]["citation_integrity_accuracy"] == 1
    assert result["threshold_failures"] == {}
    assert result["thresholds"] == json.loads(
        (root / "evals" / "thresholds.json").read_text(encoding="utf-8")
    )
