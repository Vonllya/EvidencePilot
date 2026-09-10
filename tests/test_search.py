import pytest

from evidencepilot.models import SearchResult
from evidencepilot.search import TavilySearchProvider, deduplicate_queries, deduplicate_results


def test_search_result_url_and_similar_title_deduplication():
    results = [
        SearchResult(title="A detailed research guide", url="https://example.com/a"),
        SearchResult(title="A detailed research guide", url="https://example.com/a/"),
        SearchResult(title="Independent article", url="https://example.com/b"),
    ]
    assert len(deduplicate_results(results)) == 2


def test_query_deduplication_including_completed():
    assert deduplicate_queries([" Alpha ", "alpha", "Beta"], ["beta"]) == ["Alpha"]


@pytest.mark.asyncio
async def test_tavily_uses_basic_search_result_limit_and_configured_timeout(monkeypatch):
    observed = {}

    async def wait_for(awaitable, timeout):
        observed["timeout"] = timeout
        return await awaitable

    monkeypatch.setattr("evidencepilot.retrieval.search.asyncio.wait_for", wait_for)

    class Client:
        async def search(self, **kwargs):
            assert kwargs == {"query": "checkpoint", "max_results": 2, "search_depth": "basic"}
            return {"results": [{"title": "Docs", "url": "https://docs.example.com", "content": "text"}]}

    provider = TavilySearchProvider("test-key", timeout=7.5)
    provider.client = Client()
    results = await provider.search("checkpoint", max_results=2)
    assert observed["timeout"] == 7.5
    assert len(results) == 1 and str(results[0].url).startswith("https://docs.example.com")
