"""Pure workflow routing decisions."""

from collections.abc import Mapping
from typing import Any


def evaluation_route(state: Mapping[str, Any]) -> str:
    sufficient = state.get("evaluation", {}).get("sufficient", False)
    exhausted = state.get("research_round", 0) >= state.get("max_rounds", 2)
    return "report" if sufficient or exhausted else "refine"


def resume_node(state: Mapping[str, Any], latest: Mapping[str, Any] | None) -> str:
    if not latest or latest["status"] in {"started", "failed"}:
        return str(latest["node"]) if latest else "plan_research"
    node = str(latest["node"])
    next_nodes = {
        "plan_research": "search_web", "search_web": "fetch_sources",
        "fetch_sources": "extract_evidence", "extract_evidence": "evaluate_evidence",
        "refine_queries": "search_web", "write_report": "verify_citations",
    }
    if node == "evaluate_evidence":
        return "refine_queries" if evaluation_route(state) == "refine" else "write_report"
    return next_nodes.get(node, "verify_citations")
