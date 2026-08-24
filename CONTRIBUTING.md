# Contributing to EvidencePilot

Thank you for helping make evidence-based research agents more reliable.

## Development setup

EvidencePilot requires Python 3.11 or newer and uses `uv` for reproducible environments.

```bash
git clone https://github.com/Vonllya/EvidencePilot.git
cd EvidencePilot
uv sync --all-groups
uv run pytest
```

Default tests and evaluations are deterministic, use mock providers, and must not incur API cost.
Copy `.env.example` to `.env` only when intentionally running opt-in live checks. Never commit
credentials, `.env`, generated artifacts, databases, complete reasoning content, or copied
proprietary source material.

## Making a change

1. Create a focused branch from `main`.
2. Keep provider selection explicit; never silently fall back from a real provider to a mock.
3. Add unit tests for behavior changes and fixed-corpus cases for citation-quality changes.
4. Preserve the distinction between `search_result`, `fetched`, and `snippet_fallback`.
5. Update documentation when configuration, output, security boundaries, or API cost changes.

Before opening a pull request, run:

```bash
uv run ruff check .
uv run pytest
uv run python scripts/run_evals.py
uv build
```

Live checks are opt-in and may incur cost:

```bash
uv run pytest -m live
uv run python scripts/run_evals.py --live-model
```

Do not include live outputs unless they have been reviewed for secrets and licensed content.

## Pull requests

Keep pull requests small enough to review. Explain the behavior change, tests, compatibility,
security impact, provider/API cost impact, and any remaining limitation. CI must pass before merge.
Security vulnerabilities should follow [SECURITY.md](SECURITY.md), not a public issue.

## Commit and release conventions

Use clear imperative commit messages. Releases follow Semantic Versioning and record user-visible
changes in [CHANGELOG.md](CHANGELOG.md).
