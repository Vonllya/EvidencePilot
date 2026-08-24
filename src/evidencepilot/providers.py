from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from .config import Settings
from .models import (
    CitationCheck,
    Evidence,
    EvidenceEvaluation,
    RefinedQueries,
    ResearchPlan,
    Source,
    SubQuestion,
    SubquestionAssessment,
)

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A user-safe provider error that never contains credentials or chain of thought."""


class LLMOutputTruncated(LLMError):
    pass


class LLMEmptyContent(LLMError):
    pass


class LLMProvider(ABC):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    retries: int = 0
    model: str = "fake"
    call_metrics: list[dict[str, Any]]

    @abstractmethod
    async def structured(self, prompt: str, schema: type[T], *, node: str = "structured") -> T: ...

    @abstractmethod
    async def text(self, prompt: str, *, node: str = "text") -> str: ...


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


class FakeLLMProvider(LLMProvider):
    """Deterministic offline model used by tests and the optional full Mock demo."""

    def __init__(self) -> None:
        self.model = "fake"
        self.calls = self.input_tokens = self.output_tokens = self.total_tokens = 0
        self.reasoning_tokens = self.cached_tokens = self.retries = 0
        self.call_metrics = []

    async def structured(self, prompt: str, schema: type[T], *, node: str = "structured") -> T:
        self.calls += 1
        self.input_tokens += len(prompt.split())
        self.total_tokens = self.input_tokens + self.output_tokens
        self.call_metrics.append({"node": node, "model": self.model, "finish_reason": "stop"})
        if schema is ResearchPlan:
            question = prompt.split("QUESTION:", 1)[-1].splitlines()[0].strip()
            data = ResearchPlan(
                objective=f"Build a verifiable answer to: {question}",
                subquestions=[
                    SubQuestion(question=f"What are the core concepts behind {question}?", search_queries=[f"{question} core concepts"]),
                    SubQuestion(question=f"What evidence supports {question}?", search_queries=[f"{question} evidence sources"]),
                    SubQuestion(question=f"What limitations and risks affect {question}?", search_queries=[f"{question} limitations risks"]),
                ],
            )
            return data  # type: ignore[return-value]
        if schema is RefinedQueries:
            return RefinedQueries(queries=["independent evidence agent reliability"] )  # type: ignore[return-value]
        if schema is EvidenceEvaluation:
            subquestions = re.findall(r'^SUBQUESTION: (.+)$', prompt, re.MULTILINE)
            assessments = [SubquestionAssessment(subquestion=q, sufficient=True, reason="Relevant bundled sources were found.", source_count=2) for q in subquestions]
            return EvidenceEvaluation(assessments=assessments, sufficient=True)  # type: ignore[return-value]
        if schema is CitationCheck:
            return CitationCheck(claim_id="CL1", source_ids=["S1"], claim="claim", verdict="supported", evidence_quality="fetched", reason="The source contains matching evidence.")  # type: ignore[return-value]
        raise TypeError(f"FakeLLMProvider has no response for {schema.__name__}")

    async def text(self, prompt: str, *, node: str = "text") -> str:
        self.calls += 1
        self.input_tokens += len(prompt.split())
        self.total_tokens = self.input_tokens + self.output_tokens
        self.call_metrics.append({"node": node, "model": self.model, "finish_reason": "stop"})
        if "EXTRACT_EVIDENCE" in prompt:
            return ""
        return ""

    def extract_evidence(self, sources: list[Source], subquestions: list[str]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for i, question in enumerate(subquestions):
            for source in sources[i % len(sources) : i % len(sources) + 2]:
                sentence = source.content.split(". ")[0].strip() + "."
                evidence.append(Evidence(claim=sentence, evidence=sentence, source_id=source.source_id, relevance_score=0.85, subquestion=question))
        return evidence


def load_mock_documents() -> list[dict[str, str]]:
    path = Path(__file__).resolve().parents[2] / "examples" / "mock_sources.json"
    return json.loads(path.read_text(encoding="utf-8"))
