"""Low-cost real-LLM acceptance run. Reads credentials only from environment."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from evidencepilot.app import create_runtime
from evidencepilot.config import Settings

QUESTION = (
    "LangGraph 的持久化 checkpoint 解决了什么问题？请基于公开技术资料总结，"
    "并说明它对长时间运行 Agent 的价值。"
)
ARTIFACTS = Path("artifacts")


def audit_summary(audit: dict) -> dict:
    checks = audit.get("checks", [])
    counts = {
        status: sum(1 for check in checks if check.get("verdict", check.get("status")) == status)
        for status in ("supported", "partially_supported", "unsupported")
    }
    total = len(checks)
    return {
        **counts,
        "valid_citations": audit.get("valid_citations", 0),
        "total_citations": audit.get("total_citations", 0),
        "coverage": audit.get("coverage", 0),
        "supported_rate": counts["supported"] / total if total else 0,
        "non_unsupported_rate": (counts["supported"] + counts["partially_supported"]) / total if total else 0,
    }


async def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    settings = replace(
        Settings.from_env(),
        llm_provider="deepseek",
        search_provider="tavily",
        openai_base_url="https://api.deepseek.com",
        openai_model="deepseek-v4-flash",
        llm_thinking_mode="disabled",
        llm_max_tokens_structured=min(Settings.from_env().llm_max_tokens_structured, 2048),
        llm_max_tokens_report=4096,
        max_queries_per_round=3,
        results_per_query=3,
        max_sources=5,
        database_path=str(ARTIFACTS / "acceptance.db"),
    )
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    runtime = create_runtime(settings)
    state = await asyncio.wait_for(runtime.workflow.run(QUESTION, max_rounds=1), timeout=600)
    task_id = state["task_id"]
    report_path = ARTIFACTS / f"{task_id}.md"
    summary_path = ARTIFACTS / f"{task_id}.summary.json"
    report_path.write_text(state.get("report", ""), encoding="utf-8")

    source_ids = {source["source_id"] for source in state.get("sources", [])}
    audit_ids = {
        source_id
        for check in state.get("citation_audit", {}).get("checks", [])
        for source_id in check.get("source_ids", [check.get("source_id")])
        if source_id
    }
    with sqlite3.connect(settings.database_path) as db:
        task_row = db.execute(
            "SELECT status FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        node_rows = db.execute(
            "SELECT node,status FROM node_runs WHERE task_id=? ORDER BY id", (task_id,)
        ).fetchall()
    summary = {
        "task_id": task_id,
        "mode": "full-real" if os.getenv("TAVILY_API_KEY") else "real-deepseek+mock-search",
        "llm_mode": "real",
        "search_mode": "tavily" if os.getenv("TAVILY_API_KEY") else "mock",
        "fetch_mode": "real_http",
        "model": settings.openai_model,
        "thinking_mode": settings.llm_thinking_mode,
        "question": QUESTION,
        "max_rounds": 1,
        "nodes": [{"node": node, "status": status} for node, status in node_rows],
        "checks": {
            "nonempty_plan": bool(state.get("research_plan")),
            "search_results": len(state.get("search_results", [])),
            "deduplicated_sources": len(state.get("sources", [])),
            "fetch_successes": state.get("metrics", {}).get("fetch_success_count", 0),
            "snippet_fallbacks": state.get("metrics", {}).get("snippet_fallback_count", 0),
            "fetch_failures": state.get("metrics", {}).get("fetch_failure_count", 0),
            "sources": len(state.get("sources", [])),
            "nonempty_evidence": bool(state.get("evidence")),
            "nonempty_report": bool(state.get("report")),
            "citation_audit_completed": bool(state.get("citation_audit")),
            "citation_source_ids_exist": audit_ids <= source_ids,
            "source_urls": [str(source["url"]) for source in state.get("sources", [])],
            "source_records": [
                {
                    "source_id": source["source_id"],
                    "status": source.get("source_status", "fetched"),
                    "title": source["title"],
                    "url": str(source["url"]),
                }
                for source in state.get("sources", [])
            ],
            "sqlite_status": task_row[0] if task_row else None,
            "rounds_completed": state.get("research_round"),
        },
        "metrics": state.get("metrics", {}),
        "citation_audit_summary": audit_summary(state.get("citation_audit", {})),
        "citation_revision": {
            "attempted": state.get("citation_revision", {}).get("attempted", False),
            "problem_claims": state.get("citation_revision", {}).get("problem_claims", 0),
            "patches_applied": state.get("citation_revision", {}).get("patches_applied", 0),
            "rechecked_claims": state.get("citation_revision", {}).get("rechecked_claims", 0),
            "before": audit_summary(state.get("citation_revision", {}).get("before", {})),
            "after": audit_summary(state.get("citation_revision", {}).get("after", {})),
        },
        "errors": state.get("errors", []),
        "report_path": str(report_path),
        "database_path": settings.database_path,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "metrics": summary["metrics"]}, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
