from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from ..config import Settings
from .base import LLMEmptyContent, LLMError, LLMOutputTruncated, LLMProvider, T

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str, settings: Settings | None = None) -> None:
        self.settings = settings or Settings(
            openai_api_key=api_key, openai_base_url=base_url, openai_model=model
        )
        # Retries are implemented here so metrics and limits remain observable.
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url,
            timeout=self.settings.llm_timeout_seconds, max_retries=0,
        )
        self.model = model
        self.api_key = api_key
        self.is_deepseek = "deepseek.com" in base_url.casefold() or model.casefold().startswith("deepseek-")
        self.calls = self.input_tokens = self.output_tokens = self.total_tokens = 0
        self.reasoning_tokens = self.cached_tokens = self.retries = 0
        self.call_metrics = []

    def _request_kwargs(self, node: str, *, structured: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": self.settings.token_budget(node)}
        if structured:
            kwargs["response_format"] = {"type": "json_object"}
        if self.is_deepseek:
            thinking: dict[str, Any] = {"type": self.settings.llm_thinking_mode}
            if self.settings.llm_thinking_mode == "enabled":
                kwargs["reasoning_effort"] = self.settings.llm_reasoning_effort
            kwargs["extra_body"] = {"thinking": thinking}
        return kwargs

    @staticmethod
    def _detail(value: Any, name: str) -> int:
        if value is None:
            return 0
        if isinstance(value, dict):
            return int(value.get(name) or 0)
        return int(getattr(value, name, 0) or 0)

    def _record(self, response: Any, node: str) -> tuple[str | None, str | None, bool]:
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
        reasoning = self._detail(getattr(usage, "completion_tokens_details", None), "reasoning_tokens")
        cached = self._detail(getattr(usage, "prompt_tokens_details", None), "cached_tokens")
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        finish_reason = getattr(choice, "finish_reason", None)
        has_reasoning = bool(getattr(message, "reasoning_content", None))
        self.input_tokens += prompt
        self.output_tokens += completion
        self.total_tokens += total
        self.reasoning_tokens += reasoning
        self.cached_tokens += cached
        self.call_metrics.append({
            "node": node, "model": getattr(response, "model", self.model),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "reasoning_tokens": reasoning,
            "cached_tokens": cached, "finish_reason": finish_reason,
            "thinking_enabled": self.settings.llm_thinking_mode == "enabled",
        })
        return content, finish_reason, has_reasoning

    def _safe_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, APIStatusError):
            messages = {
                401: "LLM authentication failed; check OPENAI_API_KEY.",
                402: "LLM account balance is insufficient.",
                422: "LLM parameters or model configuration are incompatible.",
                429: "LLM rate limit exceeded after finite retries.",
                500: "LLM service failed after finite retries.",
                503: "LLM service is unavailable after finite retries.",
            }
            return LLMError(messages.get(exc.status_code, f"LLM API failed with HTTP {exc.status_code}."))
        if isinstance(exc, (APITimeoutError, APIConnectionError, asyncio.TimeoutError)):
            return LLMError("LLM network request timed out or could not connect after finite retries.")
        message = str(exc).replace(self.api_key, "[REDACTED]")
        return LLMError(message)

    async def _create(self, *, prompt: str, node: str, structured: bool) -> Any:
        kwargs = self._request_kwargs(node, structured=structured)
        for attempt in range(self.settings.llm_max_retries + 1):
            self.calls += 1
            try:
                return await self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs
                )
            except Exception as exc:
                status = exc.status_code if isinstance(exc, APIStatusError) else None
                retryable = status in {429, 500, 503} or isinstance(
                    exc, (APITimeoutError, APIConnectionError, asyncio.TimeoutError)
                )
                if not retryable or attempt >= self.settings.llm_max_retries:
                    raise self._safe_error(exc) from None
                self.retries += 1
                await asyncio.sleep(min(0.5 * (2**attempt) + random.random() * 0.1, 4.0))

    async def structured(self, prompt: str, schema: type[T], *, node: str = "structured") -> T:
        error: Exception | None = None
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        repair = ""
        for parse_attempt in range(2):
            try:
                response = await self._create(
                    prompt=(prompt + "\nReturn only a JSON object matching this JSON schema: " + schema_text + repair),
                    node=node, structured=True,
                )
                content, finish_reason, has_reasoning = self._record(response, node)
                if finish_reason == "length":
                    raise LLMOutputTruncated(
                        f"{node} JSON output was truncated; increase its token budget "
                        f"(currently {self.settings.token_budget(node)})."
                    )
                if not content:
                    hint = " Reasoning consumed the output budget." if has_reasoning else ""
                    raise LLMEmptyContent(f"{node} returned empty content.{hint}")
                return schema.model_validate_json(content)
            except (LLMOutputTruncated, LLMEmptyContent) as exc:
                error = exc
                if parse_attempt == 1:
                    break
                repair = "\nThe prior response was empty or truncated. Produce a smaller complete JSON object."
            except Exception as exc:
                error = exc
                if isinstance(exc, LLMError):
                    raise
                repair = "\nThe prior response was invalid. Repair it and return only valid JSON."
        raise LLMError(f"{node} structured output failed after one repair attempt: {error}")

    async def text(self, prompt: str, *, node: str = "text") -> str:
        response = await self._create(prompt=prompt, node=node, structured=False)
        content, finish_reason, has_reasoning = self._record(response, node)
        if finish_reason == "length":
            raise LLMOutputTruncated(
                f"{node} output was truncated; increase its token budget "
                f"(currently {self.settings.token_budget(node)})."
            )
        if not content:
            hint = " Reasoning consumed the output budget." if has_reasoning else ""
            raise LLMEmptyContent(f"{node} returned empty content.{hint}")
        return content
