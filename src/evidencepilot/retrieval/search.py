from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable
from difflib import SequenceMatcher

from tavily import AsyncTavilyClient

from ..models import SearchResult
from ..providers import load_mock_documents


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 4) -> list[SearchResult]: ...


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self.client = AsyncTavilyClient(api_key=api_key)
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        try:
            response = await asyncio.wait_for(
                self.client.search(query=query, max_results=max_results, search_depth="basic"),
                timeout=self.timeout,
            )
        except TimeoutError:
            raise RuntimeError("Tavily request timed out") from None
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = f" (HTTP {status})" if status else ""
            raise RuntimeError(f"Tavily request failed{detail}") from None
        return [SearchResult(title=r.get("title", "Untitled"), url=r["url"], snippet=r.get("content", ""), query=query) for r in response.get("results", [])]


class MockSearchProvider(SearchProvider):
    def __init__(self, documents: list[dict[str, str]] | None = None) -> None:
        self.documents = documents or load_mock_documents()

    async def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        terms = set(query.casefold().split())
        ranked = sorted(self.documents, key=lambda d: len(terms & set((d["title"] + " " + d["content"]).casefold().split())), reverse=True)
        return [SearchResult(title=d["title"], url=d["url"], snippet=d["content"], query=query) for d in ranked[:max_results]]


def deduplicate_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    kept: list[SearchResult] = []
    urls: set[str] = set()
    for result in results:
        normalized = str(result.url).rstrip("/").casefold()
        if normalized in urls:
            continue
        if any(SequenceMatcher(None, result.title.casefold(), old.title.casefold()).ratio() > 0.93 for old in kept):
            continue
        urls.add(normalized)
        kept.append(result)
    return kept


def deduplicate_queries(queries: Iterable[str], completed: Iterable[str] = ()) -> list[str]:
    seen = {q.strip().casefold() for q in completed}
    output: list[str] = []
    for query in queries:
        query = query.strip()
        if query and query.casefold() not in seen:
            seen.add(query.casefold())
            output.append(query)
    return output
