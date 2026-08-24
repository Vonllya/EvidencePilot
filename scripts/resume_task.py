"""Resume a failed or interrupted EvidencePilot task from SQLite."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from evidencepilot.app import create_runtime
from evidencepilot.config import Settings

ROOT = Path(__file__).resolve().parents[1]


async def resume(task_id: str) -> None:
    load_dotenv(ROOT / ".env", override=False)
    runtime = create_runtime(Settings.from_env())
    state = await runtime.workflow.resume(task_id)
    print(json.dumps({
        "task_id": state["task_id"],
        "status": runtime.workflow.store.task_status(task_id) if runtime.workflow.store else None,
        "report_ready": bool(state.get("report")),
        "errors": len(state.get("errors", [])),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    args = parser.parse_args()
    asyncio.run(resume(args.task_id))


if __name__ == "__main__":
    main()
