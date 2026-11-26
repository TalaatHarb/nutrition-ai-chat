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
