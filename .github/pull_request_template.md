## Summary

Describe the focused behavior change and why it is needed.

## Verification

- [ ] `uv run ruff check .`
- [ ] `uv run pytest`
- [ ] `uv run python scripts/run_evals.py`
- [ ] `uv build`
- [ ] New or changed behavior has appropriate tests.
- [ ] Default tests remain offline and do not incur API cost.

## Safety and compatibility

- [ ] No credentials, `.env`, sensitive artifacts, or complete reasoning content are included.
- [ ] Provider fallback behavior remains explicit.
- [ ] Fetching, privacy, dependency, and API-cost impact have been considered.
- [ ] User-visible or configuration changes are documented.

## Remaining limitations

List known limitations or follow-up work.
