import pytest

from evidencepilot.app import create_runtime
from evidencepilot.config import Settings
from evidencepilot.fetcher import WebFetcher
from evidencepilot.models import (
    CitationAudit,
    CitationBatch,
    CitationBatchItem,
    ClaimPatch,
    ClaimPatchBatch,
    ResearchPlan,
    SearchResult,
)
from evidencepilot.providers import FakeLLMProvider, LLMProvider
from evidencepilot.search import MockSearchProvider, SearchProvider
from evidencepilot.storage import SQLiteStore
from evidencepilot.workflow import ResearchWorkflow


def test_explicit_mock_selects_mock_providers(tmp_path):
    runtime = create_runtime(Settings(
        llm_provider="mock", search_provider="mock", database_path=str(tmp_path / "test.db")
    ))
    assert isinstance(runtime.workflow.search, MockSearchProvider)
    assert runtime.search_mode == "Mock Search（显式配置）"


def test_missing_real_provider_keys_do_not_silently_fallback(tmp_path):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_runtime(Settings(database_path=str(tmp_path / "test.db")))


@pytest.mark.asyncio
async def test_mock_end_to_end_and_max_rounds():
    workflow = ResearchWorkflow(FakeLLMProvider(), MockSearchProvider(), WebFetcher())
    result = await workflow.run("How can research agents remain verifiable?", max_rounds=1)
    assert result["research_round"] == 1
    assert result["report"].startswith("# Research Report")
    assert result["sources"] and result["evidence"]
    assert CitationAudit.model_validate(result["citation_audit"]).coverage == 1


@pytest.mark.asyncio
async def test_citation_source_id_existence_check():
    workflow = ResearchWorkflow(FakeLLMProvider(), MockSearchProvider(), WebFetcher())
    state = workflow.initial_state("A sufficiently long question?")
    state["report"] = "A fabricated conclusion [S99]."
    state["sources"] = []
    audit = await workflow.verify_citations(state)
    assert audit["citation_audit"]["checks"][0]["verdict"] == "unsupported"


@pytest.mark.asyncio
async def test_failed_http_fetch_is_recorded_as_snippet_fallback():
    class RealSearchStub(SearchProvider):
        async def search(self, query: str, max_results: int = 4):
            return []

    class FailingFetcher:
        async def fetch_page(self, url: str):
            raise RuntimeError("blocked")

    workflow = ResearchWorkflow(FakeLLMProvider(), RealSearchStub(), FailingFetcher())
    state = workflow.initial_state("A sufficiently long question?")
    state["search_results"] = [SearchResult(
        title="Docs", url="https://docs.example.com", snippet="Useful fallback", query="q"
    ).model_dump(mode="json")]
    update = await workflow.fetch_sources(state)
    assert update["sources"][0]["source_status"] == "snippet_fallback"
    assert update["metrics"]["fetch_success_count"] == 0
    assert update["metrics"]["snippet_fallback_count"] == 1
    assert update["metrics"]["fetch_failure_count"] == 1


@pytest.mark.asyncio
async def test_batch_audit_revises_unsupported_claims_and_rechecks():
    class BatchLLM(LLMProvider):
        def __init__(self):
            self.calls = self.input_tokens = self.output_tokens = self.total_tokens = 0
            self.reasoning_tokens = self.cached_tokens = self.retries = 0
            self.model = "batch-stub"
            self.call_metrics = []

        async def structured(self, prompt, schema, *, node="structured"):
            self.calls += 1
            if schema is ClaimPatchBatch:
                return ClaimPatchBatch(patches=[ClaimPatch(
                    claim_id="CL1", action="replace",
                    replacement="A narrower supported claim [S1].",
                )])
            if self.calls == 1:
                return CitationBatch(checks=[
                    CitationBatchItem(claim_id="CL1", verdict="unsupported", reason="broad"),
                    CitationBatchItem(claim_id="CL2", verdict="supported", reason="already sound"),
                ])
            assert "CL2" not in prompt
            return CitationBatch(checks=[CitationBatchItem(
                claim_id="CL1", verdict="supported", reason="batch decision"
            )])

        async def text(self, prompt, *, node="text"):
            self.calls += 1
            raise AssertionError("local patch revision must use structured output")

    llm = BatchLLM()
    workflow = ResearchWorkflow(llm, MockSearchProvider(), WebFetcher())
    state = workflow.initial_state("A sufficiently long question?")
    state["report"] = "An unsupported broad claim [S1]. A supported fact [S1].\n\n## References\n\n- [S1] [Docs](https://docs.example.com/)"
    state["sources"] = [{
        "source_id": "S1", "title": "Docs", "url": "https://docs.example.com/",
        "content": "A narrower supported claim. A supported fact.", "source_status": "fetched",
    }]
    update = await workflow.verify_citations(state)
    assert llm.calls == 3
    assert update["citation_revision"]["attempted"] is True
    assert update["citation_revision"]["problem_claims"] == 1
    assert update["citation_revision"]["patches_applied"] == 1
    assert update["citation_revision"]["rechecked_claims"] == 1
    assert update["citation_audit"]["checks"][0]["verdict"] == "supported"
    assert "narrower supported claim" in update["report"]
    assert "A supported fact [S1]." in update["report"]


@pytest.mark.asyncio
async def test_failed_workflow_resumes_from_failed_node(tmp_path):
    class FailPlanOnce(FakeLLMProvider):
        def __init__(self):
            super().__init__()
            self.failed = False

        async def structured(self, prompt, schema, *, node="structured"):
            if schema is ResearchPlan and not self.failed:
                self.failed = True
                raise RuntimeError("transient plan failure")
            return await super().structured(prompt, schema, node=node)

    store = SQLiteStore(str(tmp_path / "resume.db"))
    workflow = ResearchWorkflow(
        FailPlanOnce(), MockSearchProvider(), WebFetcher(), store=store,
        settings=Settings(llm_provider="mock", search_provider="mock"),
    )
    with pytest.raises(RuntimeError, match="transient plan failure"):
        await workflow.run("How can a persisted workflow resume safely?", max_rounds=1)
    task = store.list_tasks()[0]
    assert task["status"] == "failed"
    assert store.latest_node_run(task["task_id"])["node"] == "plan_research"

    resumed = await workflow.resume(task["task_id"])
    assert resumed["report"].startswith("# Research Report")
    assert store.task_status(task["task_id"]) == "completed"
    completed = await workflow.resume(task["task_id"])
    assert completed["task_id"] == task["task_id"]
