from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from copy import deepcopy
from typing import Any

import httpx
from langgraph.graph import END, START, StateGraph

from ..citations.audit import CitationAuditMixin
from ..config import Settings
from ..models import (
    Evidence,
    EvidenceBatch,
    EvidenceEvaluation,
    Metrics,
    RefinedQueries,
    ResearchPlan,
    ResearchState,
    SearchResult,
    Source,
)
from ..observability import log_event
from ..providers import FakeLLMProvider, LLMProvider
from ..retrieval.fetch import WebFetcher
from ..retrieval.search import (
    MockSearchProvider,
    SearchProvider,
    deduplicate_queries,
    deduplicate_results,
)
from ..storage import StateStore
from .routing import evaluation_route, resume_node


class WorkflowNodes(CitationAuditMixin):
    NODE_NAMES = (
        "plan_research", "search_web", "fetch_sources", "extract_evidence",
        "evaluate_evidence", "refine_queries", "write_report", "verify_citations",
    )

    def __init__(self, llm: LLMProvider, search: SearchProvider, fetcher: WebFetcher, store: StateStore | None = None, settings: Settings | None = None) -> None:
        self.llm, self.search, self.fetcher, self.store = llm, search, fetcher, store
        self.settings = settings or Settings()
        self._node_lock = asyncio.Lock()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("resume_router", self._resume_router)
        for name in self.NODE_NAMES:
            graph.add_node(name, self._wrapped(name, getattr(self, name)))
        graph.add_edge(START, "resume_router")
        graph.add_conditional_edges(
            "resume_router", self._route_start, {name: name for name in self.NODE_NAMES}
        )
        graph.add_edge("plan_research", "search_web")
        graph.add_edge("search_web", "fetch_sources")
        graph.add_edge("fetch_sources", "extract_evidence")
        graph.add_edge("extract_evidence", "evaluate_evidence")
        graph.add_conditional_edges("evaluate_evidence", self._route_evaluation, {"refine": "refine_queries", "report": "write_report"})
        graph.add_edge("refine_queries", "search_web")
        graph.add_edge("write_report", "verify_citations")
        graph.add_edge("verify_citations", END)
        return graph.compile()

    async def _resume_router(self, state: ResearchState) -> dict[str, Any]:
        return {"resume_from": state.get("resume_from", "plan_research")}

    async def _route_start(self, state: ResearchState) -> str:
        node = state.get("resume_from", "plan_research")
        return node if node in self.NODE_NAMES else "plan_research"

    def _wrapped(self, name: str, function):
        async def run(state: ResearchState) -> dict[str, Any]:
            node_started = time.monotonic()
            before = self._provider_metrics()
            calls_before = self.llm.calls
            logger = logging.getLogger("evidencepilot.workflow")
            log_event(logger, "node_started", task_id=state["task_id"], node=name)
            if self.store:
                self.store.record_node(state["task_id"], name, "started")
            try:
                update = await function(state)
                merged = deepcopy(state)
                merged.update(update)
                self._update_metrics(
                    merged, name, time.monotonic() - node_started, before
                )
                update["metrics"] = merged["metrics"]
                if self.store:
                    for message in merged.get("errors", [])[len(state.get("errors", [])):]:
                        self.store.record_error(state["task_id"], name, message)
                    self.store.save_state(merged, "completed" if name == "verify_citations" else "running")
                    self.store.record_node(state["task_id"], name, "completed")
                log_event(
                    logger, "node_completed", task_id=state["task_id"], node=name,
                    elapsed_seconds=round(time.monotonic() - node_started, 3),
                    model_calls=self.llm.calls - calls_before,
                )
                return update
            except Exception as exc:
                message = f"{name}: {type(exc).__name__}: {exc}"
                if self.store:
                    self.store.record_node(state["task_id"], name, "failed", message)
                    self.store.record_error(state["task_id"], name, message)
                    failed = deepcopy(state)
                    self._update_metrics(failed, name, time.monotonic() - node_started, before)
                    self.store.save_state(failed, "failed")
                log_event(
                    logger, "node_failed", task_id=state["task_id"], node=name,
                    error_type=type(exc).__name__,
                )
                raise
        async def serialized(state: ResearchState) -> dict[str, Any]:
            # A runtime shares one provider; isolate each node's usage deltas.
            async with self._node_lock:
                return await run(state)
        return serialized

    def initial_state(
        self,
        question: str,
        max_rounds: int = 2,
        source_preference: str = "不限定来源类型",
        max_sources: int | None = None,
    ) -> ResearchState:
        now = time.time()
        source_limit = self.settings.max_sources if max_sources is None else max(1, min(max_sources, 50))
        return ResearchState(task_id=str(uuid.uuid4()), question=question.strip(), research_plan={}, queries=[], completed_queries=[], search_results=[], sources=[], evidence=[], missing_information=[], research_round=0, max_rounds=max(1, min(max_rounds, 3)), source_preference=source_preference.strip() or "不限定来源类型", max_sources=source_limit, report="", citation_audit={}, citation_revision={}, errors=[], metrics=Metrics(started_at=now).model_dump())

    def _provider_metrics(self) -> dict[str, int]:
        return {
            "model_calls": self.llm.calls, "prompt_tokens": self.llm.input_tokens,
            "completion_tokens": self.llm.output_tokens, "total_tokens": self.llm.total_tokens,
            "reasoning_tokens": self.llm.reasoning_tokens, "cached_tokens": self.llm.cached_tokens,
            "retry_count": self.llm.retries, "call_count": len(self.llm.call_metrics),
        }

    def _update_metrics(
        self, state: dict[str, Any], node: str, elapsed: float, before: dict[str, int]
    ) -> None:
        metrics = deepcopy(state.get("metrics", {}))
        current = self._provider_metrics()
        node_calls = current["model_calls"] - before["model_calls"]
        node_metrics = dict(metrics.get("node_metrics", {}))
        previous = node_metrics.get(node, {})
        node_metrics[node] = {
            "calls": previous.get("calls", 0) + 1,
            "model_calls": previous.get("model_calls", 0) + node_calls,
            "elapsed_seconds": round(previous.get("elapsed_seconds", 0) + elapsed, 3),
        }
        for key, value in current.items():
            if key != "call_count":
                metrics[key] = metrics.get(key, 0) + value - before[key]
        metrics.update(
            elapsed_seconds=round(time.time() - metrics.get("started_at", time.time()), 3),
            model=self.llm.model, node_metrics=node_metrics,
            call_metrics=[*metrics.get("call_metrics", []),
                          *self.llm.call_metrics[before["call_count"]:]],
        )
        state["metrics"] = metrics

    async def plan_research(self, state: ResearchState) -> dict[str, Any]:
        plan = await self.llm.structured(f"Create a research plan with 3-5 subquestions. Prefer source type: {state.get('source_preference', '不限定来源类型')}. QUESTION: {state['question']}\nReturn only JSON.", ResearchPlan, node="plan_research")
        queries = [q for sub in plan.subquestions for q in sub.search_queries]
        return {"research_plan": plan.model_dump(mode="json"), "queries": deduplicate_queries(queries)[: self.settings.max_queries_per_round]}

    async def search_web(self, state: ResearchState) -> dict[str, Any]:
        queries = deduplicate_queries(state.get("queries", []), state.get("completed_queries", []))[: self.settings.max_queries_per_round]
        batches = await asyncio.gather(*(self.search.search(q, self.settings.results_per_query) for q in queries), return_exceptions=True)
        errors = list(state.get("errors", []))
        results = []
        for query, batch in zip(queries, batches, strict=True):
            if isinstance(batch, Exception):
                errors.append(f"search failed for {query}: {batch}")
            else:
                results.extend(batch)
        # Reserve source capacity for gap-driven searches in later rounds.
        limit = state.get("max_sources", self.settings.max_sources)
        round_number = state.get("research_round", 0) + 1
        rounds = state.get("max_rounds", 2)
        budget = min(limit, (limit * round_number + rounds - 1) // rounds)
        combined = deduplicate_results([*map(SearchResult.model_validate, state.get("search_results", [])), *results])[:budget]
        metrics = dict(state["metrics"])
        metrics["search_count"] = metrics.get("search_count", 0) + len(queries)
        return {"search_results": [r.model_dump(mode="json") for r in combined], "completed_queries": [*state.get("completed_queries", []), *queries], "research_round": state.get("research_round", 0) + 1, "errors": errors, "metrics": metrics}

    async def fetch_sources(self, state: ResearchState) -> dict[str, Any]:
        previous = {s["url"]: Source.model_validate(s) for s in state.get("sources", [])}
        pending = [r for r in state["search_results"] if r["url"] not in previous]

        async def fetch_one(result):
            if isinstance(self.search, MockSearchProvider):
                return result.get("snippet", ""), "snippet_fallback", None, None
            try:
                page = await self.fetcher.fetch_page(result["url"])
                if not page.content.strip():
                    raise ValueError("parsed page body was empty")
                return page.content, "fetched", None, page
            except Exception as exc:
                return "", "", exc, None

        def retryable_fetch_error(error: Exception) -> bool:
            if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException, httpx.NetworkError)):
                return True
            if isinstance(error, httpx.HTTPStatusError):
                return error.response.status_code in {408, 425, 429, 500, 502, 503, 504}
            return False

        fetched = await asyncio.gather(*(fetch_one(r) for r in pending), return_exceptions=True)
        retry_candidates = [
            (index, result, outcome)
            for index, (result, outcome) in enumerate(zip(pending, fetched, strict=True))
            if not isinstance(outcome, Exception)
            and outcome[2] is not None
            and retryable_fetch_error(outcome[2])
        ]
        if retry_candidates:
            await asyncio.sleep(0.2)
            retries = await asyncio.gather(
                *(fetch_one(result) for _, result, _ in retry_candidates), return_exceptions=True
            )
            for (index, _, _), retry_outcome in zip(retry_candidates, retries, strict=True):
                fetched[index] = retry_outcome

        errors = list(state.get("errors", []))
        sources = list(previous.values())
        for result, outcome in zip(pending, fetched, strict=True):
            if isinstance(outcome, Exception):
                errors.append(f"fetch failed for {result['url']}: {outcome}")
                continue
            content, status, failure, page = outcome
            if failure is not None:
                errors.append(f"fetch failed for {result['url']}: {type(failure).__name__}: {failure}")
                snippet = result.get("snippet", "").strip()
                content, status, page = snippet, "snippet_fallback" if snippet else "search_result", None
            if content.strip():
                sources.append(Source(
                    source_id=f"S{len(sources)+1}", title=result["title"], url=result["url"],
                    content=content[:self.settings.max_source_chars], query=result.get("query", ""),
                    source_status=status, final_url=page.final_url if page else result["url"],
                    http_status=page.http_status if page else None,
                    content_type=page.content_type if page else "tavily/snippet",
                    retrieved_at=page.retrieved_at if page else "",
                    content_hash=page.content_hash if page else "",
                    parser=page.parser if page else "tavily",
                    quality_score=0.9 if status == "fetched" else 0.4,
                ))
        metrics = dict(state["metrics"])
        metrics["fetch_count"] = metrics.get("fetch_count", 0) + len(pending)
        metrics["fetch_attempt_count"] = metrics.get("fetch_attempt_count", 0) + len(pending) + len(retry_candidates)
        metrics["fetch_retry_count"] = metrics.get("fetch_retry_count", 0) + len(retry_candidates)
        metrics["timeout_retry_count"] = metrics.get("timeout_retry_count", 0) + sum(
            isinstance(outcome[2], (asyncio.TimeoutError, httpx.TimeoutException))
            for _, _, outcome in retry_candidates
        )
        statuses = [source.source_status for source in sources]
        metrics["fetch_success_count"] = statuses.count("fetched")
        metrics["snippet_fallback_count"] = statuses.count("snippet_fallback")
        metrics["fetch_failure_count"] = sum(1 for error in errors if error.startswith("fetch failed for "))
        return {"sources": [s.model_dump(mode="json") for s in sources[:state.get("max_sources", self.settings.max_sources)]], "errors": errors, "metrics": metrics}

    async def extract_evidence(self, state: ResearchState) -> dict[str, Any]:
        sources = [Source.model_validate(s) for s in state.get("sources", [])]
        questions = [s["question"] for s in state["research_plan"]["subquestions"]]
        if not sources:
            return {
                "evidence": [],
                "errors": [*state.get("errors", []), "No readable sources were available."],
            }
        if isinstance(self.llm, FakeLLMProvider):
            items = self.llm.extract_evidence(sources, questions)
        else:
            allowed = [s.source_id for s in sources]
            prompt = "Extract concise evidence for every subquestion. Use only allowed source_id values.\nALLOWED: " + json.dumps(allowed) + "\nSUBQUESTIONS: " + json.dumps(questions) + "\nSOURCES: " + json.dumps([s.model_dump(mode="json") for s in sources])
            batch = await self.llm.structured(prompt + "\nReturn only JSON.", EvidenceBatch, node="extract_evidence")
            items = [e for e in batch.evidence if e.source_id in allowed]
        seen = set()
        unique = []
        for item in [*map(Evidence.model_validate, state.get("evidence", [])), *items]:
            key = (item.source_id, item.claim.casefold(), item.subquestion.casefold())
            if key not in seen:
                seen.add(key)
                unique.append(item.model_dump())
        return {"evidence": unique}

    async def evaluate_evidence(self, state: ResearchState) -> dict[str, Any]:
        questions = [s["question"] for s in state["research_plan"]["subquestions"]]
        lines = "\n".join(f"SUBQUESTION: {q}" for q in questions)
        evaluation = await self.llm.structured(f"Assess sufficiency and source concentration per question. Return only JSON.\n{lines}\nEVIDENCE: {json.dumps(state.get('evidence', []))}", EvidenceEvaluation, node="evaluate_evidence")
        return {"evaluation": evaluation.model_dump(), "missing_information": evaluation.missing_information}

    @staticmethod
    def _evaluation_route(state: ResearchState) -> str:
        return evaluation_route(state)

    async def _route_evaluation(self, state: ResearchState) -> str:
        return self._evaluation_route(state)

    async def refine_queries(self, state: ResearchState) -> dict[str, Any]:
        missing = state.get("missing_information") or [a["subquestion"] for a in state.get("evaluation", {}).get("assessments", []) if not a["sufficient"]]
        refined = await self.llm.structured("Generate supplemental queries only for these gaps, avoiding completed queries. Return only JSON. GAPS: " + json.dumps(missing) + " COMPLETED: " + json.dumps(state.get("completed_queries", [])), RefinedQueries, node="refine_queries")
        return {"queries": deduplicate_queries(refined.queries, state.get("completed_queries", []))[:self.settings.max_queries_per_round]}

    async def write_report(self, state: ResearchState) -> dict[str, Any]:
        sources = [Source.model_validate(s) for s in state.get("sources", [])]
        evidence = state.get("evidence", [])
        findings = "\n".join(f"- {e['claim']} [{e['source_id']}]" for e in evidence) or "- No reliable evidence was extracted."
        refs = "\n".join(f"- [{s.source_id}] [{s.title}]({s.url})" for s in sources)
        if isinstance(self.llm, FakeLLMProvider):
            report = f"# Research Report: {state['question']}\n\n## Executive Summary\n\nThe findings below synthesize the collected and traceable evidence.\n\n## Research Process\n\nEvidencePilot planned {len(state['research_plan']['subquestions'])} subquestions, executed {len(state.get('completed_queries', []))} searches across {state.get('research_round', 0)} round(s), and retained {len(sources)} sources.\n\n## Key Findings\n\n{findings}\n\n## Limitations\n\nSearch coverage is bounded and source quality varies.\n\n## References\n\n{refs}"
        else:
            prompt = (
                "Write a concise Markdown research report using only the supplied evidence. "
                "Write atomic factual claims: each sentence must contain only one independently "
                "verifiable fact. Put its supporting source IDs immediately before that sentence's "
                "ending punctuation, exactly like '... [S1].' Never attach one citation group to "
                "multiple facts or an entire paragraph. "
                "Do not invent source IDs. Include Executive Summary, Key Findings, Limitations, "
                "Conclusion, and References with clickable Markdown links.\nQUESTION: "
                + state["question"] + "\nEVIDENCE: " + json.dumps(evidence, ensure_ascii=False)
                + "\nSOURCES: " + json.dumps([
                    {"source_id": s.source_id, "title": s.title, "url": str(s.url)} for s in sources
                ], ensure_ascii=False)
            )
            report = await self.llm.text(prompt, node="write_report")
        return {"report": report}

    async def run(self, question: str, max_rounds: int = 2) -> ResearchState:
        state = self.initial_state(question, max_rounds)
        if self.store:
            self.store.save_state(dict(state), "running")
        return await self.graph.ainvoke(state, {"recursion_limit": 30})

    def _resume_node(self, state: ResearchState, latest: dict[str, Any] | None) -> str:
        return resume_node(state, latest)

    async def resume(self, task_id: str) -> ResearchState:
        if not self.store:
            raise RuntimeError("resume requires a SQLiteStore")
        state = self.store.load_state(task_id)
        if state is None:
            raise KeyError(f"task not found: {task_id}")
        if self.store.task_status(task_id) == "completed":
            return state  # type: ignore[return-value]
        state["resume_from"] = self._resume_node(state, self.store.latest_node_run(task_id))
        self.store.save_state(state, "running")
        return await self.graph.ainvoke(state, {"recursion_limit": 30})
