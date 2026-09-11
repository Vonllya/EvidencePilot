import httpx
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


class EmptySearch(SearchProvider):
    async def search(self, query: str, max_results: int = 4):
        return []


def test_initial_state_preserves_frontend_research_controls():
    workflow = ResearchWorkflow(FakeLLMProvider(), MockSearchProvider(), WebFetcher())
    state = workflow.initial_state(
        "A sufficiently long question?",
        max_rounds=3,
        source_preference="论文平台",
        max_sources=6,
    )
    assert state["max_rounds"] == 3
    assert state["source_preference"] == "论文平台"
    assert state["max_sources"] == 6


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
    audit = CitationAudit.model_validate(result["citation_audit"])
    assert 0 < audit.coverage < 1
    assert any(not check.source_ids for check in audit.checks)


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
async def test_timeout_is_retried_after_first_fetch_pass():
    from evidencepilot.fetcher import FetchedPage

    class RetryingFetcher:
        def __init__(self):
            self.calls = []

        async def fetch_page(self, url: str):
            self.calls.append(url)
            if len(self.calls) == 1:
                raise httpx.ReadTimeout("temporary timeout")
            return FetchedPage(
                content="Recovered evidence", final_url=url, http_status=200,
                content_type="text/plain", retrieved_at="now", content_hash="hash",
                parser="text",
            )

    fetcher = RetryingFetcher()
    workflow = ResearchWorkflow(FakeLLMProvider(), EmptySearch(), fetcher)
    state = workflow.initial_state("A sufficiently long question?")
    state["search_results"] = [SearchResult(
        title="Docs", url="https://docs.example.com", snippet="fallback", query="q"
    ).model_dump(mode="json")]
    update = await workflow.fetch_sources(state)
    assert len(fetcher.calls) == 2
    assert update["sources"][0]["source_status"] == "fetched"
    assert update["metrics"]["fetch_retry_count"] == 1
    assert update["metrics"]["timeout_retry_count"] == 1
    assert update["metrics"]["fetch_failure_count"] == 0


@pytest.mark.asyncio
async def test_http_403_is_not_retried():
    class ForbiddenFetcher:
        def __init__(self):
            self.calls = 0

        async def fetch_page(self, url: str):
            self.calls += 1
            request = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("forbidden", request=request, response=httpx.Response(403, request=request))

    fetcher = ForbiddenFetcher()
    workflow = ResearchWorkflow(FakeLLMProvider(), EmptySearch(), fetcher)
    state = workflow.initial_state("A sufficiently long question?")
    state["search_results"] = [SearchResult(
        title="Docs", url="https://docs.example.com", snippet="fallback", query="q"
    ).model_dump(mode="json")]
    update = await workflow.fetch_sources(state)
    assert fetcher.calls == 1
    assert update["sources"][0]["source_status"] == "snippet_fallback"
    assert update["metrics"]["fetch_retry_count"] == 0


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
            assert "CL2" in prompt
            return CitationBatch(checks=[
                CitationBatchItem(claim_id="CL1", verdict="supported", reason="rechecked"),
                CitationBatchItem(claim_id="CL2", verdict="supported", reason="rechecked"),
            ])

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
    assert update["citation_revision"]["rechecked_claims"] == 2
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


def test_delete_task_removes_task_and_dependent_records(tmp_path):
    store = SQLiteStore(str(tmp_path / "delete.db"))
    state = ResearchWorkflow(FakeLLMProvider(), MockSearchProvider(), WebFetcher()).initial_state("Delete me")
    state["sources"] = [{
        "source_id": "S1", "title": "Docs", "url": "https://docs.example.com",
        "content": "evidence", "source_status": "fetched",
    }]
    store.save_state(state)
    store.record_node(state["task_id"], "plan_research", "completed")
    store.record_error(state["task_id"], "fetch_sources", "temporary error")
    assert store.delete_task(state["task_id"]) is True
    assert store.load_state(state["task_id"]) is None
    assert store.delete_task(state["task_id"]) is False


@pytest.mark.asyncio
async def test_later_rounds_add_sources_within_total_budget():
    class RoundSearch(MockSearchProvider):
        async def search(self, query, max_results=4):
            return [SearchResult(title=f"{query} {i}", url=f"https://example.com/{query}/{i}",
                                 snippet="Useful evidence.") for i in range(10)]

    workflow = ResearchWorkflow(FakeLLMProvider(), RoundSearch(), WebFetcher())
    state = workflow.initial_state("Research with evidence gaps", max_rounds=3, max_sources=10)
    for round_number, expected_count in enumerate([4, 7, 10], start=1):
        previous = list(state["sources"])
        state["queries"] = [f"round{round_number}"]
        state.update(await workflow.search_web(state))
        state.update(await workflow.fetch_sources(state))
        assert len(state["sources"]) == expected_count
        assert state["sources"][:len(previous)] == previous
        assert any(f"/round{round_number}/" in s["url"] for s in state["sources"])
    assert len({s["source_id"] for s in state["sources"]}) == 10


@pytest.mark.asyncio
async def test_reused_runtime_keeps_task_metrics_separate():
    workflow = ResearchWorkflow(FakeLLMProvider(), MockSearchProvider(), WebFetcher())
    first = await workflow.run("How can research remain verifiable?")
    second = await workflow.run("How can research remain verifiable?")
    for key in ("model_calls", "prompt_tokens", "total_tokens", "call_metrics"):
        assert first["metrics"][key] == second["metrics"][key]
    assert second["metrics"]["model_calls"] == sum(
        node["model_calls"] for node in second["metrics"]["node_metrics"].values()
    )


@pytest.mark.asyncio
async def test_resume_with_new_provider_preserves_failed_attempt_usage(tmp_path):
    class FailEvaluation(FakeLLMProvider):
        async def structured(self, prompt, schema, *, node="structured"):
            result = await super().structured(prompt, schema, node=node)
            if node == "evaluate_evidence":
                raise RuntimeError("evaluation failed")
            return result

    store = SQLiteStore(str(tmp_path / "metrics.db"))
    workflow = ResearchWorkflow(FailEvaluation(), MockSearchProvider(), WebFetcher(), store)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        await workflow.run("How can research remain verifiable?", max_rounds=1)
    task_id = store.list_tasks()[0]["task_id"]
    before = store.load_state(task_id)["metrics"]
    assert before["model_calls"] == 2
    resumed_workflow = ResearchWorkflow(FakeLLMProvider(), MockSearchProvider(), WebFetcher(), store)
    result = await resumed_workflow.resume(task_id)
    after = result["metrics"]
    assert after["model_calls"] == 3
    assert after["total_tokens"] > before["total_tokens"]
    assert after["call_metrics"][:len(before["call_metrics"])] == before["call_metrics"]
    assert after["node_metrics"]["evaluate_evidence"]["calls"] == 2


def test_full_source_budget_stops_unproductive_supplemental_searches():
    from evidencepilot.workflow.routing import evaluation_route

    assert evaluation_route({
        "evaluation": {"sufficient": False}, "research_round": 1,
        "max_rounds": 3, "max_sources": 1, "search_results": [{"url": "https://example.com"}],
    }) == "report"


@pytest.mark.asyncio
async def test_concurrent_tasks_on_shared_runtime_keep_usage_separate():
    import asyncio

    class YieldingLLM(FakeLLMProvider):
        async def structured(self, prompt, schema, *, node="structured"):
            result = await super().structured(prompt, schema, node=node)
            await asyncio.sleep(0)
            return result

    workflow = ResearchWorkflow(YieldingLLM(), MockSearchProvider(), WebFetcher())
    first, second = await asyncio.gather(
        workflow.run("How can research remain verifiable?"),
        workflow.run("How can research remain verifiable?"),
    )
    assert first["metrics"]["model_calls"] == second["metrics"]["model_calls"] == 2
    assert first["metrics"]["total_tokens"] == second["metrics"]["total_tokens"]
