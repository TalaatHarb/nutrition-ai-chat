from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: object
    id: str | None = None


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    model: str
    message: LLMMessage


@dataclass(frozen=True)
class TextDelta:
    content: str


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
