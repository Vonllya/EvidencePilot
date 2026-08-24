import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import load_dotenv

from evidencepilot.config import Settings
from evidencepilot.search import TavilySearchProvider

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_real_tavily_online_availability():
    if os.getenv("RUN_LIVE_TAVILY") != "1":
        pytest.skip("set RUN_LIVE_TAVILY=1 to call the real Tavily API")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    settings = Settings(
        llm_provider="mock", search_provider="tavily",
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY is not configured")
    provider = TavilySearchProvider(settings.tavily_api_key)
    results = await provider.search("LangGraph checkpoint official documentation", max_results=2)
    assert results
    assert all(urlparse(str(result.url)).scheme in {"http", "https"} for result in results)
    assert all(urlparse(str(result.url)).hostname != "example.org" for result in results)
