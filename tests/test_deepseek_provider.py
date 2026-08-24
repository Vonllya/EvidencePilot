from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from evidencepilot.config import Settings
from evidencepilot.models import CitationDecision
from evidencepilot.providers import LLMError, OpenAILLMProvider


def response(
    content: str | None,
    *,
    finish: str = "stop",
    reasoning: str | None = None,
    reasoning_tokens: int | None = 3,
):
    completion_details = (
        SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
    )
    return SimpleNamespace(
        model="deepseek-v4-flash",
        choices=[SimpleNamespace(
            finish_reason=finish,
            message=SimpleNamespace(content=content, reasoning_content=reasoning),
        )],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            completion_tokens_details=completion_details,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        ),
    )


class StubCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.kwargs = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def provider(settings: Settings, replies=()):
    instance = OpenAILLMProvider("secret-test-key", settings.openai_base_url, settings.openai_model, settings)
    stub = StubCompletions(replies)
    instance.client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return instance, stub


def test_deepseek_disabled_thinking_extra_body():
    llm, _ = provider(Settings(openai_base_url="https://api.deepseek.com", openai_model="deepseek-v4-flash"))
    kwargs = llm._request_kwargs("plan_research", structured=True)
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in kwargs


def test_deepseek_enabled_passes_reasoning_effort():
    settings = Settings(openai_base_url="https://api.deepseek.com", openai_model="deepseek-v4-flash", llm_thinking_mode="enabled", llm_reasoning_effort="low")
    llm, _ = provider(settings)
    kwargs = llm._request_kwargs("plan_research", structured=True)
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kwargs["reasoning_effort"] == "low"


def test_non_deepseek_has_no_vendor_parameters():
    llm, _ = provider(Settings(), ())
    kwargs = llm._request_kwargs("plan_research", structured=True)
    assert "extra_body" not in kwargs and "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_reasoning_content_is_not_business_content_and_usage_is_optional():
    llm, _ = provider(Settings(openai_base_url="https://api.deepseek.com", openai_model="deepseek-v4-flash"), [response("FINAL", reasoning="private chain", reasoning_tokens=None)])
    assert await llm.text("hello") == "FINAL"
    assert llm.reasoning_tokens == 0
    assert "private chain" not in str(llm.call_metrics)


@pytest.mark.asyncio
async def test_length_empty_is_truncated_and_structured_retries_once():
    good = '{"status":"supported","reason":"valid evidence"}'
    llm, stub = provider(Settings(openai_base_url="https://api.deepseek.com", openai_model="deepseek-v4-flash"), [response(None, finish="length", reasoning="budget used"), response(good)])
    result = await llm.structured("Return JSON", CitationDecision, node="verify_citations")
    assert result.status == "supported" and len(stub.kwargs) == 2


@pytest.mark.asyncio
async def test_second_invalid_json_safely_fails():
    llm, stub = provider(Settings(), [response("{"), response("not json")])
    with pytest.raises(LLMError, match="after one repair attempt"):
        await llm.structured("Return JSON", CitationDecision)
    assert len(stub.kwargs) == 2


def test_api_key_is_redacted_from_errors_and_logs(caplog):
    llm, _ = provider(Settings())
    with caplog.at_level(logging.DEBUG):
        safe = llm._safe_error(RuntimeError("failed secret-test-key"))
    assert "secret-test-key" not in str(safe)
    assert "secret-test-key" not in caplog.text


def test_node_token_budgets():
    settings = Settings(llm_max_tokens_structured=2048, llm_max_tokens_report=4096)
    assert settings.token_budget("plan_research") == 2048
    assert settings.token_budget("extract_evidence") == 2048
    assert settings.token_budget("evaluate_evidence") == 1536
    assert settings.token_budget("refine_queries") == 1024
    assert settings.token_budget("verify_citations") == 2048
    assert settings.token_budget("write_report") == 4096
    assert settings.token_budget("revise_unsupported_claims") == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("replies", [
    [APITimeoutError(request=httpx.Request("POST", "https://example.com"))] * 3,
])
async def test_retryable_errors_have_finite_retries(monkeypatch, replies):
    settings = Settings(llm_max_retries=2)
    llm, stub = provider(settings, replies)

    async def no_sleep(_):
        return None

    monkeypatch.setattr("evidencepilot.providers.asyncio.sleep", no_sleep)
    with pytest.raises(LLMError, match="finite retries"):
        await llm.text("hello")
    assert len(stub.kwargs) == 3 and llm.retries == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_retryable_http_statuses_have_finite_retries(monkeypatch, status):
    request = httpx.Request("POST", "https://example.com/chat/completions")
    errors = [APIStatusError(
        f"HTTP {status}", response=httpx.Response(status, request=request), body=None
    ) for _ in range(3)]
    llm, stub = provider(Settings(llm_max_retries=2), errors)

    async def no_sleep(_):
        return None

    monkeypatch.setattr("evidencepilot.providers.asyncio.sleep", no_sleep)
    with pytest.raises(LLMError, match="finite retries"):
        await llm.text("hello")
    assert len(stub.kwargs) == 3 and llm.retries == 2
