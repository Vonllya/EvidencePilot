from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .fetcher import WebFetcher
from .providers import FakeLLMProvider, LLMProvider, OpenAILLMProvider
from .search import MockSearchProvider, SearchProvider, TavilySearchProvider
from .storage import SQLiteStore
from .workflow import ResearchWorkflow


@dataclass(slots=True)
class Runtime:
    workflow: ResearchWorkflow
    llm_mode: str
    search_mode: str
    runnable: bool


def create_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or Settings.from_env()
    llm: LLMProvider
    if settings.llm_provider == "mock":
        llm, llm_mode = FakeLLMProvider(), "Mock LLM（显式配置）"
    else:
        if not settings.openai_api_key:
            raise ValueError(
                f"LLM_PROVIDER={settings.llm_provider} requires OPENAI_API_KEY; "
                "use LLM_PROVIDER=mock explicitly for offline mode"
            )
        llm = OpenAILLMProvider(
            settings.openai_api_key, settings.openai_base_url, settings.openai_model, settings
        )
        llm_mode = f"{settings.llm_provider} ({settings.openai_model})"
    search: SearchProvider
    if settings.search_provider == "mock":
        search, search_mode = MockSearchProvider(), "Mock Search（显式配置）"
    else:
        if not settings.tavily_api_key:
            raise ValueError(
                "SEARCH_PROVIDER=tavily requires TAVILY_API_KEY; "
                "use SEARCH_PROVIDER=mock explicitly for offline mode"
            )
        search, search_mode = TavilySearchProvider(settings.tavily_api_key), "Tavily"
    workflow = ResearchWorkflow(
        llm, search,
        WebFetcher(settings.http_timeout_seconds, settings.max_page_bytes, settings.max_source_chars),
        SQLiteStore(settings.database_path), settings,
    )
    return Runtime(workflow, llm_mode, search_mode, True)
