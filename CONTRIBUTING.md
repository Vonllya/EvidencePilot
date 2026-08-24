# Contributing

1. Create a branch from `main`.
2. Run `uv sync --all-groups`.
3. Keep default tests offline and deterministic.
4. Add fixed-corpus coverage for citation changes.
5. Run `uv run ruff check .`, `uv run pytest`, `uv run python scripts/run_evals.py`, and `uv build`.

Live tests must remain opt-in and clearly identify API cost. Never commit `.env`, credentials, generated artifacts, databases, complete reasoning content, or copied proprietary source material.

Pull requests should explain behavior changes, tests, security impact, provider cost impact, and compatibility considerations.
