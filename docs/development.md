# Development

Use Python 3.11+ and uv:

```bash
uv sync --all-groups
LLM_PROVIDER=mock SEARCH_PROVIDER=mock uv run evidencepilot providers
uv run pytest
uv run python scripts/run_evals.py
```

Real providers require explicit selection and credentials. Default tests and fixed evaluations must never call external APIs. Use the `live` marker only for explicit connectivity checks.

The CLI provides `research`, `resume`, `inspect`, `eval`, and `providers`. Structured JSON logs contain task/node identifiers, duration, model-call counts, and error types, but not prompts, source bodies, credentials, or reasoning content.
