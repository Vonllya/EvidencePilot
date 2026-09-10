"""Provider contracts and user-safe errors."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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

__all__ = ["LLMEmptyContent", "LLMError", "LLMOutputTruncated", "LLMProvider"]
