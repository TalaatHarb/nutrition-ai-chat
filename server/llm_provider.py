from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from llm_models import LLMMessage, LLMResponse, TextDelta, ToolDefinition


class ProviderError(Exception):
    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class LLMProvider(ABC):
    @abstractmethod
    async def health(self) -> bool:
        """Check provider availability without generating a response."""

    @abstractmethod
    async def complete(
        self, messages: list[LLMMessage], tools: list[ToolDefinition], model: str | None = None,
    ) -> LLMResponse:
        """Generate one assistant turn, which may request tools."""

    @abstractmethod
    def stream(
        self, messages: list[LLMMessage], tools: list[ToolDefinition], model: str | None = None,
    ) -> AsyncIterator[TextDelta | LLMResponse]:
        """Yield text deltas followed by one complete turn, including any tool calls."""
