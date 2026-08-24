"""CLI wrapper for the shared citation evaluation implementation."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from evidencepilot.evals import DEFAULT_CORPUS, DEFAULT_THRESHOLDS, evaluate_citations

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--live-model", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    result = asyncio.run(evaluate_citations(args.corpus, args.live_model, args.thresholds))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["threshold_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
