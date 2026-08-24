from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from markdown_it import MarkdownIt

from .config import Settings
from .fetcher import WebFetcher
from .models import (
    AtomicClaim,
    CitationAudit,
    CitationBatch,
    CitationCheck,
    ClaimPatchBatch,
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
from .observability import log_event
from .providers import FakeLLMProvider, LLMProvider
from .search import MockSearchProvider, SearchProvider, deduplicate_queries, deduplicate_results
from .storage import SQLiteStore


class ResearchWorkflow:
    NODE_NAMES = (
        "plan_research", "search_web", "fetch_sources", "extract_evidence",
        "evaluate_evidence", "refine_queries", "write_report", "verify_citations",
    )

    def __init__(self, llm: LLMProvider, search: SearchProvider, fetcher: WebFetcher, store: SQLiteStore | None = None, settings: Settings | None = None) -> None:
        self.llm, self.search, self.fetcher, self.store = llm, search, fetcher, store
        self.settings = settings or Settings()
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
            calls_before = self.llm.calls
            logger = logging.getLogger("evidencepilot.workflow")
            log_event(logger, "node_started", task_id=state["task_id"], node=name)
            if self.store:
                self.store.record_node(state["task_id"], name, "started")
            try:
                update = await function(state)
                merged = dict(state)
                merged.update(update)
                self._update_metrics(
                    merged, name, time.monotonic() - node_started, self.llm.calls - calls_before
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
                    self.store.save_state(dict(state), "failed")
                log_event(
                    logger, "node_failed", task_id=state["task_id"], node=name,
                    error_type=type(exc).__name__,
                )
                raise
        return run

    def initial_state(self, question: str, max_rounds: int = 2) -> ResearchState:
        now = time.time()
        return ResearchState(task_id=str(uuid.uuid4()), question=question.strip(), research_plan={}, queries=[], completed_queries=[], search_results=[], sources=[], evidence=[], missing_information=[], research_round=0, max_rounds=max(1, min(max_rounds, 3)), report="", citation_audit={}, citation_revision={}, errors=[], metrics=Metrics(started_at=now).model_dump())

    def _update_metrics(
        self, state: dict[str, Any], node: str, elapsed: float, node_calls: int
    ) -> None:
        metrics = state.get("metrics", {})
        node_metrics = dict(metrics.get("node_metrics", {}))
        previous = node_metrics.get(node, {})
        node_metrics[node] = {
            "calls": previous.get("calls", 0) + 1,
            "model_calls": previous.get("model_calls", 0) + node_calls,
            "elapsed_seconds": round(previous.get("elapsed_seconds", 0) + elapsed, 3),
        }
        metrics.update(
            elapsed_seconds=round(time.time() - metrics.get("started_at", time.time()), 3),
            model_calls=self.llm.calls, prompt_tokens=self.llm.input_tokens,
            completion_tokens=self.llm.output_tokens, total_tokens=self.llm.total_tokens,
            reasoning_tokens=self.llm.reasoning_tokens, cached_tokens=self.llm.cached_tokens,
            retry_count=self.llm.retries, model=self.llm.model, node_metrics=node_metrics,
            call_metrics=list(self.llm.call_metrics),
        )

    async def plan_research(self, state: ResearchState) -> dict[str, Any]:
        plan = await self.llm.structured(f"Create a research plan with 3-5 subquestions. QUESTION: {state['question']}\nReturn only JSON.", ResearchPlan, node="plan_research")
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
        combined = deduplicate_results([*map(SearchResult.model_validate, state.get("search_results", [])), *results])[: self.settings.max_sources]
        metrics = dict(state["metrics"])
        metrics["search_count"] = metrics.get("search_count", 0) + len(queries)
        return {"search_results": [r.model_dump(mode="json") for r in combined], "completed_queries": [*state.get("completed_queries", []), *queries], "research_round": state.get("research_round", 0) + 1, "errors": errors, "metrics": metrics}

    async def fetch_sources(self, state: ResearchState) -> dict[str, Any]:
        previous = {s["url"]: Source.model_validate(s) for s in state.get("sources", [])}
        pending = [r for r in state["search_results"] if r["url"] not in previous]
        async def fetch_one(result):
            if isinstance(self.search, MockSearchProvider):
                return result.get("snippet", ""), "snippet_fallback", None, None
            last = None
            for _ in range(2):
                try:
                    page = await self.fetcher.fetch_page(result["url"])
                    if not page.content.strip():
                        raise ValueError("parsed page body was empty")
                    return page.content, "fetched", None, page
                except Exception as exc:
                    last = exc
            snippet = result.get("snippet", "").strip()
            return snippet, "snippet_fallback" if snippet else "search_result", last, None
        fetched = await asyncio.gather(*(fetch_one(r) for r in pending), return_exceptions=True)
        errors = list(state.get("errors", []))
        sources = list(previous.values())
        for result, outcome in zip(pending, fetched, strict=True):
            if isinstance(outcome, Exception):
                errors.append(f"fetch failed for {result['url']}: {outcome}")
                continue
            content, status, failure, page = outcome
            if failure is not None:
                errors.append(f"fetch failed for {result['url']}: {type(failure).__name__}: {failure}")
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
        statuses = [source.source_status for source in sources]
        metrics["fetch_success_count"] = statuses.count("fetched")
        metrics["snippet_fallback_count"] = statuses.count("snippet_fallback")
        metrics["fetch_failure_count"] = sum(1 for error in errors if error.startswith("fetch failed for "))
        return {"sources": [s.model_dump(mode="json") for s in sources[:self.settings.max_sources]], "errors": errors, "metrics": metrics}

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
        sufficient = state.get("evaluation", {}).get("sufficient", False)
        return "report" if sufficient or state.get("research_round", 0) >= state.get("max_rounds", 2) else "refine"

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

    @staticmethod
    def _atomic_claims(report: str) -> list[AtomicClaim]:
        claims: list[AtomicClaim] = []
        parser = MarkdownIt("commonmark").enable("table")
        tokens = parser.parse(report)
        lines = report.splitlines(keepends=True)
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        stack: list[str] = []
        references_level: int | None = None
        heading_level: int | None = None
        heading_text = ""
        reference_names = {"references", "reference", "参考文献", "参考资料", "来源", "sources"}

        for token in tokens:
            if token.type == "heading_open":
                heading_level = int(token.tag[1:])
                stack.append(token.type)
                continue
            if token.type == "heading_close":
                normalized = heading_text.strip().rstrip(":：").casefold()
                if normalized in reference_names:
                    references_level = heading_level
                elif references_level is not None and heading_level is not None and heading_level <= references_level:
                    references_level = None
                heading_level = None
                heading_text = ""
                if stack and stack[-1] == "heading_open":
                    stack.pop()
                continue
            if token.type == "tr_open" and "tbody_open" in stack and token.map is not None:
                start_line, end_line = token.map
                block_start = line_offsets[start_line]
                block_end = line_offsets[min(end_line, len(line_offsets) - 1)]
                raw_row = report[block_start:block_end]
                row_text = raw_row.strip().strip("|").strip()
                source_ids = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", row_text)))
                if source_ids:
                    local_start = raw_row.find(row_text)
                    start_offset = block_start + local_start
                    end_offset = start_offset + len(row_text)
                    claims.append(AtomicClaim(
                        claim_id=f"CL{len(claims) + 1}", text=row_text,
                        source_ids=source_ids, node_type="table_row",
                        start_offset=start_offset, end_offset=end_offset,
                        start_line=start_line, end_line=max(start_line, end_line - 1),
                    ))
            if token.nesting == 1:
                stack.append(token.type)
                continue
            if token.nesting == -1:
                opening = token.type.removesuffix("_close") + "_open"
                if opening in stack:
                    stack.remove(opening)
                continue
            if token.type != "inline":
                continue
            if "td_open" in stack or "th_open" in stack:
                continue
            if "heading_open" in stack:
                heading_text = token.content
                continue
            if references_level is not None or token.map is None:
                continue

            start_line, end_line = token.map
            block_start = line_offsets[start_line]
            block_end = line_offsets[min(end_line, len(line_offsets) - 1)]
            raw_block = report[block_start:block_end]
            inline_start = raw_block.find(token.content)
            if inline_start < 0:
                continue
            cursor = 0
            sentences = re.split(
                r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z0-9*_`])", token.content
            )
            if "td_open" in stack or "th_open" in stack:
                node_type = "table_cell"
            elif "list_item_open" in stack:
                node_type = "list_item"
            elif "blockquote_open" in stack:
                node_type = "blockquote"
            else:
                node_type = "paragraph"
            for sentence_raw in sentences:
                sentence = sentence_raw.strip()
                if not sentence:
                    continue
                local_start = token.content.find(sentence, cursor)
                if local_start < 0:
                    continue
                cursor = local_start + len(sentence)
                source_ids = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", sentence)))
                if not source_ids:
                    continue
                start_offset = block_start + inline_start + local_start
                end_offset = start_offset + len(sentence)
                claims.append(AtomicClaim(
                    claim_id=f"CL{len(claims) + 1}", text=sentence,
                    source_ids=source_ids, node_type=node_type,
                    start_offset=start_offset, end_offset=end_offset,
                    start_line=report.count("\n", 0, start_offset),
                    end_line=report.count("\n", 0, end_offset),
                ))
        return claims

    @staticmethod
    def _evidence_quality(source_ids: list[str], source_map: dict[str, dict[str, Any]]) -> str:
        if not source_ids or any(source_id not in source_map for source_id in source_ids):
            return "unavailable"
        statuses = {source_map[source_id].get("source_status", "fetched") for source_id in source_ids}
        if statuses == {"fetched"}:
            return "fetched"
        if statuses == {"snippet_fallback"}:
            return "snippet_fallback"
        return "mixed"

    async def _batch_citation_audit(
        self, claims: list[AtomicClaim], source_map: dict[str, dict[str, Any]]
    ) -> CitationAudit:
        checks: list[CitationCheck] = []
        requests: list[dict[str, str]] = []
        valid = 0
        for claim in claims:
            quality = self._evidence_quality(claim.source_ids, source_map)
            if quality == "unavailable":
                checks.append(CitationCheck(
                    claim_id=claim.claim_id, source_ids=claim.source_ids, claim=claim.text,
                    verdict="unsupported", evidence_quality=quality,
                    reason="At least one cited source_id does not exist.",
                ))
                continue
            valid += 1
            requests.append({
                "claim_id": claim.claim_id,
                "source_ids": claim.source_ids,
                "claim": claim.text,
            })
        if isinstance(self.llm, FakeLLMProvider):
            checks.extend(CitationCheck(
                claim_id=item["claim_id"], source_ids=item["source_ids"], claim=item["claim"],
                verdict="supported",
                evidence_quality=self._evidence_quality(item["source_ids"], source_map),
                reason="The cited bundled source contains the extracted passage.",
            ) for item in requests)
        elif requests:
            cited_ids = sorted({source_id for item in requests for source_id in item["source_ids"]})
            source_payload = [{
                "source_id": source_id,
                "content": source_map[source_id]["content"],
            } for source_id in cited_ids]
            prompt = (
                "Audit sentence-level atomic claims in one batch. For each claim_id, judge whether its "
                "cited source_ids AS A SET jointly support the complete claim. A source need not support "
                "the entire claim individually. Use supported only when the source set supports every "
                "material part, partially_supported for incomplete or broader-than-source claims, and "
                "unsupported for no material support or contradiction. Return exactly one check per "
                "claim_id. Evidence transport quality is evaluated separately; do not use it to change "
                "the semantic verdict. Do not rewrite claims.\nCLAIMS: "
                + json.dumps(requests, ensure_ascii=False)
                + "\nSOURCES: " + json.dumps(source_payload, ensure_ascii=False)
            )
            batch = await self.llm.structured(
                prompt + "\nReturn only JSON.", CitationBatch, node="verify_citations"
            )
            decisions = {item.claim_id: item for item in batch.checks}
            for item in requests:
                decision = decisions.get(item["claim_id"])
                checks.append(CitationCheck(
                    claim_id=item["claim_id"], source_ids=item["source_ids"], claim=item["claim"],
                    verdict=decision.verdict if decision else "unsupported",
                    evidence_quality=self._evidence_quality(item["source_ids"], source_map),
                    reason=decision.reason if decision else "The batch audit omitted this citation.",
                ))
        order = {claim.claim_id: index for index, claim in enumerate(claims)}
        checks.sort(key=lambda check: order.get(check.claim_id, len(order)))
        total = len(claims)
        return CitationAudit(
            checks=checks, valid_citations=valid, total_citations=total,
            coverage=valid / total if total else 0,
        )

    @staticmethod
    def _apply_claim_patches(
        report: str, patch_batch: ClaimPatchBatch, problem_map: dict[str, AtomicClaim],
        source_map: dict[str, dict[str, Any]],
    ) -> tuple[str, int]:
        edits: list[tuple[int, int, str]] = []
        used_claims: set[str] = set()
        for patch in patch_batch.patches:
            claim = problem_map.get(patch.claim_id)
            if not claim or patch.claim_id in used_claims:
                continue
            if report[claim.start_offset:claim.end_offset] != claim.text:
                continue
            replacement = patch.replacement.strip() if patch.action == "replace" else ""
            replacement_ids = re.findall(r"\[(S\d+)\]", replacement)
            if patch.action == "replace" and (
                not replacement_ids or any(source_id not in source_map for source_id in replacement_ids)
            ):
                continue
            edits.append((claim.start_offset, claim.end_offset, replacement))
            used_claims.add(patch.claim_id)
        for start_offset, end_offset, replacement in sorted(edits, reverse=True):
            report = report[:start_offset] + replacement + report[end_offset:]
        return report, len(edits)

    async def verify_citations(self, state: ResearchState) -> dict[str, Any]:
        source_map = {s["source_id"]: s for s in state.get("sources", [])}
        report = state.get("report", "")
        original_claims = self._atomic_claims(report)
        before = await self._batch_citation_audit(original_claims, source_map)
        problems = [check for check in before.checks if check.verdict != "supported"]
        revision = {
            "attempted": False, "problem_claims": len(problems), "rechecked_claims": 0,
            "patches_applied": 0,
            "before": before.model_dump(), "after": before.model_dump(),
        }
        if problems and not isinstance(self.llm, FakeLLMProvider):
            cited_ids = sorted(source_map)
            prompt = (
                "Create local structured patches only for the listed problem atomic claims. Use replace "
                "to narrow a claim to one independently verifiable sentence with citations immediately "
                "next to the supported fact, or delete when no support exists. Do not edit supported "
                "claims or other report text. Replacement source IDs must come from ALLOWED SOURCES.\n"
                "PROBLEM CLAIMS:\n" + json.dumps([c.model_dump() for c in problems], ensure_ascii=False)
                + "\nALLOWED SOURCES:\n" + json.dumps([{
                    "source_id": source_id, "title": source_map[source_id]["title"],
                    "url": str(source_map[source_id]["url"]),
                    "content": source_map[source_id]["content"],
                } for source_id in cited_ids], ensure_ascii=False)
            )
            patch_batch = await self.llm.structured(
                prompt + "\nReturn only JSON.", ClaimPatchBatch, node="revise_unsupported_claims"
            )
            problem_map = {claim.claim_id: claim for claim in original_claims if claim.claim_id in {c.claim_id for c in problems}}
            report, applied = self._apply_claim_patches(
                report, patch_batch, problem_map, source_map
            )
            final_claims = self._atomic_claims(report)
            before_by_text = {check.claim: check for check in before.checks}
            changed_claims = [claim for claim in final_claims if claim.text not in before_by_text]
            changed_audit = await self._batch_citation_audit(changed_claims, source_map)
            changed_by_text = {check.claim: check for check in changed_audit.checks}
            final_checks = [before_by_text.get(claim.text) or changed_by_text[claim.text] for claim in final_claims]
            valid = sum(check.evidence_quality != "unavailable" for check in final_checks)
            after = CitationAudit(
                checks=final_checks, valid_citations=valid, total_citations=len(final_checks),
                coverage=valid / len(final_checks) if final_checks else 0,
            )
            revision.update(
                attempted=True, patches_applied=applied, rechecked_claims=len(changed_claims),
                after=after.model_dump(),
            )
            return {
                "report": report, "citation_audit": after.model_dump(),
                "citation_revision": revision,
            }
        return {"citation_audit": before.model_dump(), "citation_revision": revision}

    async def run(self, question: str, max_rounds: int = 2) -> ResearchState:
        state = self.initial_state(question, max_rounds)
        if self.store:
            self.store.save_state(dict(state), "running")
        return await self.graph.ainvoke(state, {"recursion_limit": 30})

    def _resume_node(self, state: ResearchState, latest: dict[str, Any] | None) -> str:
        if not latest or latest["status"] in {"started", "failed"}:
            return latest["node"] if latest else "plan_research"
        node = latest["node"]
        next_nodes = {
            "plan_research": "search_web",
            "search_web": "fetch_sources",
            "fetch_sources": "extract_evidence",
            "extract_evidence": "evaluate_evidence",
            "refine_queries": "search_web",
            "write_report": "verify_citations",
        }
        if node == "evaluate_evidence":
            return "refine_queries" if self._evaluation_route(state) == "refine" else "write_report"
        return next_nodes.get(node, "verify_citations")

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
