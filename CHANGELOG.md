# Changelog

All notable changes to EvidencePilot are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- CodeQL, Dependabot, ownership rules, and structured contribution templates.
- Fixed-corpus regression thresholds for claim parsing, citation adjacency and integrity.

### Security

- Stricter URL, redirect, port, proxy, media-type, and response-length validation.

## [0.1.0] - 2026-08-24

### Added

- LangGraph research workflow with planning, search, fetching, evidence extraction,
  sufficiency evaluation, report writing, and batched citation verification.
- Explicit DeepSeek/OpenAI-compatible, Tavily, and deterministic mock providers.
- Sentence-level Markdown AST citation parsing and structured local claim patches.
- SQLite task persistence, resumable runs, node metrics, and acceptance artifacts.
- Offline tests and a fixed citation-quality corpus, with opt-in live provider checks.
- CLI, Streamlit interface, Docker packaging, CI, security policy, and architecture docs.

### Security

- Credentials are environment-only and generated artifacts are excluded from Git.
- HTTP fetching rejects non-public targets, revalidates redirects, and limits response size.

[0.1.0]: https://github.com/Vonllya/EvidencePilot/releases/tag/v0.1.0
