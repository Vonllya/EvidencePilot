from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from .app import create_runtime
from .config import Settings
from .evals import evaluate_citations
from .observability import configure_logging


def _runtime():
    load_dotenv(Path.cwd() / ".env", override=False)
    return create_runtime(Settings.from_env())


async def _run(args) -> None:
    if args.command == "research":
        state = await _runtime().workflow.run(args.question, args.max_rounds)
        print(json.dumps({"task_id": state["task_id"], "report": state.get("report", "")}, ensure_ascii=False, indent=2))
    elif args.command == "resume":
        runtime = _runtime()
        state = await runtime.workflow.resume(args.task_id)
        print(json.dumps({"task_id": state["task_id"], "status": runtime.workflow.store.task_status(args.task_id)}, ensure_ascii=False))
    elif args.command == "inspect":
        runtime = _runtime()
        state = runtime.workflow.store.load_state(args.task_id)
        if state is None:
            raise SystemExit(f"task not found: {args.task_id}")
        print(json.dumps({"task_id": args.task_id, "status": runtime.workflow.store.task_status(args.task_id), "question": state.get("question"), "report_ready": bool(state.get("report"))}, ensure_ascii=False, indent=2))
    elif args.command == "eval":
        load_dotenv(Path.cwd() / ".env", override=False)
        result = await evaluate_citations(args.corpus, args.live_model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "providers":
        load_dotenv(Path.cwd() / ".env", override=False)
        settings = Settings.from_env()
        print(json.dumps({"llm_provider": settings.llm_provider, "search_provider": settings.search_provider, "llm_key": "configured" if settings.openai_api_key else "missing", "search_key": "configured" if settings.tavily_api_key else "missing"}, indent=2))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="evidencepilot")
    sub = parser.add_subparsers(dest="command", required=True)
    research = sub.add_parser("research")
    research.add_argument("question")
    research.add_argument("--max-rounds", type=int, default=2)
    resume = sub.add_parser("resume")
    resume.add_argument("task_id")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("task_id")
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("--corpus", type=Path, default=Path("evals/citation_cases.jsonl"))
    evaluation.add_argument("--live-model", action="store_true")
    sub.add_parser("providers")
    asyncio.run(_run(parser.parse_args()))
