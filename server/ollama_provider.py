import json
import logging
import os
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from llm_provider import (
    LLMMessage, LLMProvider, LLMResponse, ProviderError, TextDelta, ToolCall, ToolDefinition,
)

logger = logging.getLogger("uvicorn.error.ollama")


class _FunctionCall(BaseModel):
    name: str
    arguments: object = None


class _ToolCall(BaseModel):
    function: _FunctionCall
    id: str | None = None


class _Message(BaseModel):
    role: str = "assistant"
    content: str = ""
    thinking: str | None = None
    tool_calls: list[_ToolCall] | None = None


class _CompletionMessage(_Message):
    role: str
    content: str


class _Completion(BaseModel):
    model: str
    message: _CompletionMessage


def _status_hint(status: int) -> str:
    return {
        401: "authentication rejected; check the API key",
        403: "access denied; check account and model access",
        404: "not found; check the endpoint and exact model name",
        429: "rate or usage limit reached",
    }.get(status, "upstream HTTP error")


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        api_key: str = "",
    ):
        self.url = url
        self.default_model = model
        self.api_key = api_key.strip()

    @classmethod
    def from_env(cls) -> "OllamaProvider":
        return cls(
            url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            api_key=os.getenv("OLLAMA_API_KEY", ""),
        )

    def endpoint(self, resource: str) -> str:
        base = self.url.strip().rstrip("/").removesuffix("/api")
        return f"{base}/api/{resource}"

    def _client(self, timeout: float | None) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        return httpx.AsyncClient(timeout=timeout, headers=headers)

    def _log_failure(
        self, endpoint: str, operation: str, reason: str,
        *, model: str | None = None, status: int | None = None,
    ) -> None:
        try:
            url = str(httpx.URL(endpoint).copy_with(userinfo=None, query=None, fragment=None))
        except httpx.InvalidURL:
            url = "<invalid URL>"
        if self.api_key:
            url = url.replace(self.api_key, "[redacted]")
            if model:
                model = model.replace(self.api_key, "[redacted]")
        logger.warning(
            "Ollama request failed: operation=%s endpoint=%r model=%r status=%s api_key_configured=%s reason=%s",
            operation, url, model[:160] if model else None, status, bool(self.api_key), reason,
        )

    def _status_error(self, status: int, endpoint: str, operation: str, model: str | None = None) -> ProviderError:
        self._log_failure(endpoint, operation, _status_hint(status), model=model, status=status)
        return ProviderError(f"Ollama rejected the request (HTTP {status}).", status)

    @staticmethod
    def _serialize_message(message: LLMMessage) -> dict:
        result: dict[str, object] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                    **({"id": call.id} if call.id else {}),
                }
                for call in message.tool_calls
            ]
        if message.tool_name is not None:
            result["tool_name"] = message.tool_name
        if message.reasoning is not None:
            result["thinking"] = message.reasoning
        return result

    @staticmethod
    def _parse_message(message: _Message) -> LLMMessage:
        return LLMMessage(
            role="assistant",
            content=message.content,
            reasoning=message.thinking,
            tool_calls=[
                ToolCall(name=call.function.name, arguments=call.function.arguments, id=call.id)
                for call in message.tool_calls or []
            ],
        )

    def _payload(self, messages: list[LLMMessage], tools: list[ToolDefinition], model: str | None, stream: bool) -> dict:
        return {
            "model": model or self.default_model,
            "messages": [self._serialize_message(message) for message in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
            "stream": stream,
        }

    async def health(self) -> bool:
        endpoint = self.endpoint("version")
        try:
            async with self._client(timeout=2) as client:
                response = await client.get(endpoint)
                if not response.is_success:
                    status = response.status_code
                    self._log_failure(endpoint, "health", _status_hint(status), status=status)
                return response.is_success
        except (httpx.HTTPError, httpx.InvalidURL) as error:
            self._log_failure(endpoint, "health", type(error).__name__)
            return False

    async def complete(
        self, messages: list[LLMMessage], tools: list[ToolDefinition], model: str | None = None,
    ) -> LLMResponse:
        endpoint = self.endpoint("chat")
        model = model or self.default_model
        payload = self._payload(messages, tools, model, False)
        try:
            async with self._client(timeout=None) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = _Completion.model_validate(response.json())
                message = data.message
                if message.role != "assistant":
                    self._log_failure(endpoint, "chat", "unexpected upstream message role", model=model)
                    raise ProviderError("Ollama returned an unexpected message role.")
                return LLMResponse(model=data.model, message=self._parse_message(message))
        except httpx.HTTPStatusError as error:
            raise self._status_error(error.response.status_code, endpoint, "chat", model) from error
        except (httpx.HTTPError, httpx.InvalidURL) as error:
            self._log_failure(endpoint, "chat", type(error).__name__, model=model)
            raise ProviderError("Ollama unreachable.") from error
        except (ValueError, KeyError, TypeError) as error:
            self._log_failure(endpoint, "chat", f"invalid upstream response ({type(error).__name__})", model=model)
            raise ProviderError("Ollama returned an invalid response.") from error

    async def stream(
        self, messages: list[LLMMessage], tools: list[ToolDefinition], model: str | None = None,
    ) -> AsyncIterator[TextDelta | LLMResponse]:
        endpoint = self.endpoint("chat")
        model = model or self.default_model
        payload = self._payload(messages, tools, model, True)
        content_parts = []
        thinking_parts = []
        tool_calls = []
        response_model = model
        try:
            async with self._client(timeout=None) as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            self._log_failure(endpoint, "chat-stream", "upstream error event", model=model)
                            raise ProviderError("Ollama could not complete the reply.")
                        delta = _Message.model_validate(chunk.get("message", {}))
                        if delta.role == "assistant":
                            tool_calls.extend(delta.tool_calls or [])
                            if delta.thinking:
                                thinking_parts.append(delta.thinking)
                            if delta.content:
                                content_parts.append(delta.content)
                                if not tool_calls:
                                    yield TextDelta(delta.content)
                        if chunk.get("done"):
                            response_model = chunk.get("model", model)
                            break
                    else:
                        self._log_failure(endpoint, "chat-stream", "upstream stream ended early", model=model)
                        raise ProviderError("Ollama ended the reply early.")
            yield LLMResponse(
                model=response_model,
                message=self._parse_message(_Message(
                    content="".join(content_parts),
                    thinking="".join(thinking_parts) or None,
                    tool_calls=tool_calls,
                )),
            )
        except httpx.HTTPStatusError as error:
            raise self._status_error(error.response.status_code, endpoint, "chat-stream", model) from error
        except (httpx.HTTPError, httpx.InvalidURL) as error:
            self._log_failure(endpoint, "chat-stream", type(error).__name__, model=model)
            raise ProviderError("Ollama unreachable.") from error
        except (ValueError, KeyError, TypeError) as error:
            self._log_failure(endpoint, "chat-stream", f"invalid upstream response ({type(error).__name__})", model=model)
            raise ProviderError("Ollama returned an invalid response.") from error
