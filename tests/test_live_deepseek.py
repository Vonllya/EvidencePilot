import os

import pytest

from evidencepilot.config import Settings
from evidencepilot.models import CitationDecision
from evidencepilot.providers import OpenAILLMProvider


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_deepseek_minimal_call():
    """Opt-in, small paid call: RUN_LIVE_TESTS=1 uv run pytest -m live."""
    if os.getenv("RUN_LIVE_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("set RUN_LIVE_TESTS=1 and OPENAI_API_KEY; this test incurs API cost")
    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
        search_provider="mock",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
        openai_model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
    )
    llm = OpenAILLMProvider(
        settings.openai_api_key or "", settings.openai_base_url, settings.openai_model, settings
    )
    decision = await llm.structured(
        "Return JSON saying this literal claim is supported: 1 equals 1.",
        CitationDecision,
        node="verify_citations",
    )
    assert decision.status in {"supported", "partially_supported"}
