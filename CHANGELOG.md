# Changelog

All notable changes to EvidencePilot are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Reserve source capacity across research rounds so gap-driven searches can add evidence.
- Keep model usage metrics task-local, including failed attempts and resumed tasks.
- Include offline mock documents in the installed package.
- Preserve independent same-title sources and case-sensitive URL paths and queries.

## [0.2.0] - 2026-09-10

### Added

- CodeQL, Dependabot, ownership rules, and structured contribution templates.
- Fixed-corpus regression thresholds for claim parsing, citation adjacency and integrity.
- Expanded deterministic citation evaluation to 32 cases and separated semantic thresholds.
- Configurable Tavily search timeout through `SEARCH_TIMEOUT_SECONDS`.
- Modular workflow, citation, retrieval, provider, and storage packages.

### Fixed

- Updated the test toolchain to pytest 9.1.1 and pytest-asyncio 1.4.0, including a refreshed lockfile.
- Citation coverage now includes uncited claims, and post-patch audits no longer match by text.

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
[0.2.0]: https://github.com/Vonllya/EvidencePilot/releases/tag/v0.2.0
