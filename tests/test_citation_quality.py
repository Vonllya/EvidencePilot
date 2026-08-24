import json
from pathlib import Path

import pytest

from evidencepilot.fetcher import WebFetcher
from evidencepilot.models import CitationBatch, CitationBatchItem, ClaimPatch, ClaimPatchBatch
from evidencepilot.providers import LLMProvider
from evidencepilot.search import MockSearchProvider
from evidencepilot.workflow import ResearchWorkflow


class FixedCorpusLLM(LLMProvider):
    def __init__(self, expected):
        self.expected = expected
        self.calls = self.input_tokens = self.output_tokens = self.total_tokens = 0
        self.reasoning_tokens = self.cached_tokens = self.retries = 0
        self.model = "fixed-corpus"
        self.call_metrics = []

    async def structured(self, prompt, schema, *, node="structured"):
        self.calls += 1
        return CitationBatch(checks=[CitationBatchItem(
            claim_id=item["claim_id"], verdict=item["verdict"], reason="fixed verdict"
        ) for item in self.expected])

    async def text(self, prompt, *, node="text"):
        raise AssertionError("quality corpus audit must not rewrite text")


@pytest.mark.asyncio
async def test_fixed_corpus_atomic_claim_verdict_and_quality_are_separate():
    fixture = Path(__file__).parent / "fixtures" / "citation_quality.json"
    corpus = json.loads(fixture.read_text(encoding="utf-8"))
    workflow = ResearchWorkflow(FixedCorpusLLM(corpus["expected"]), MockSearchProvider(), WebFetcher())
    claims = workflow._atomic_claims(corpus["report"])
    assert [(claim.claim_id, claim.source_ids) for claim in claims] == [
        ("CL1", ["S1"]), ("CL2", ["S2"])
    ]
    audit = await workflow._batch_citation_audit(
        claims, {source["source_id"]: source for source in corpus["sources"]}
    )
    actual = [{
        "claim_id": check.claim_id,
        "verdict": check.verdict,
        "evidence_quality": check.evidence_quality,
    } for check in audit.checks]
    assert actual == corpus["expected"]


def test_atomic_claim_uses_joint_source_set():
    claims = ResearchWorkflow._atomic_claims(
        "State is saved and later restored [S1][S2]. Another fact [S3]."
    )
    assert claims[0].source_ids == ["S1", "S2"]
    assert claims[1].source_ids == ["S3"]


def test_markdown_ast_respects_structure_and_source_offsets():
    report = """# Report

Paragraph fact [S1].

- List fact [S1].

> Quoted fact [S2].

```python
ignored = "not a claim [S9]."
```

## References

- [S1] [Docs](https://docs.example.com)
- [S2] [Other](https://other.example.com)
"""
    claims = ResearchWorkflow._atomic_claims(report)
    assert [(claim.text, claim.node_type) for claim in claims] == [
        ("Paragraph fact [S1].", "paragraph"),
        ("List fact [S1].", "list_item"),
        ("Quoted fact [S2].", "blockquote"),
    ]
    assert all(report[claim.start_offset:claim.end_offset] == claim.text for claim in claims)
    assert all("S9" not in claim.source_ids for claim in claims)


def test_ast_offsets_patch_only_target_duplicate_claim():
    report = "Repeated fact [S1]. Repeated fact [S1]."
    claims = ResearchWorkflow._atomic_claims(report)
    patched, applied = ResearchWorkflow._apply_claim_patches(
        report,
        ClaimPatchBatch(patches=[ClaimPatch(
            claim_id="CL2", action="replace", replacement="Changed second fact [S1]."
        )]),
        {"CL2": claims[1]},
        {"S1": {"source_id": "S1"}},
    )
    assert applied == 1
    assert patched == "Repeated fact [S1]. Changed second fact [S1]."


def test_markdown_ast_combines_table_fact_and_citation_cells():
    report = """| Claim | Evidence |
|---|---|
| State is restored | [S1] |
"""
    claims = ResearchWorkflow._atomic_claims(report)
    assert len(claims) == 1
    assert claims[0].node_type == "table_row"
    assert claims[0].text == "State is restored | [S1]"
    assert report[claims[0].start_offset:claims[0].end_offset] == claims[0].text
