from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatResponse(BaseModel):
    model: str
    message: AssistantMessage


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider_available: bool = Field(serialization_alias="ollama")
