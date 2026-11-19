import json
import logging
import os
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nutrition_tools import TOOLS, run_tool
from prompt_filter import looks_like_prompt_injection

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
SYSTEM_PROMPT = Path(__file__).with_name("nutrition_prompt.txt").read_text(encoding="utf-8").strip()
MAX_TOOL_ROUNDS = 4
MAX_CALLS_PER_ROUND = 8
FILTERED_MESSAGES_HEADER = "X-Filtered-Message-Indices"
logger = logging.getLogger("uvicorn.error.ollama")

app = FastAPI(
    title="Nutrition Assistant",
    description="Nutrition chat powered by a local Ollama model.",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[FILTERED_MESSAGES_HEADER],
)


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
    ollama: bool


def ollama_client(timeout: float | None) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None
    return httpx.AsyncClient(timeout=timeout, headers=headers)


def ollama_endpoint(resource: str) -> str:
    base = OLLAMA_URL.strip().rstrip("/").removesuffix("/api")
    return f"{base}/api/{resource}"


def log_ollama_failure(
    endpoint: str,
    operation: str,
    reason: str,
    *,
    model: str | None = None,
    status: int | None = None,
) -> None:
    try:
        url = str(httpx.URL(endpoint).copy_with(userinfo=None, query=None, fragment=None))
    except httpx.InvalidURL:
        url = "<invalid URL>"
    if OLLAMA_API_KEY:
        url = url.replace(OLLAMA_API_KEY, "[redacted]")
        if model:
            model = model.replace(OLLAMA_API_KEY, "[redacted]")
    logger.warning(
        "Ollama request failed: operation=%s endpoint=%r model=%r status=%s api_key_configured=%s reason=%s",
        operation, url, model[:160] if model else None, status, bool(OLLAMA_API_KEY), reason,
    )


def status_hint(status: int) -> str:
    return {
        401: "authentication rejected; check the API key",
        403: "access denied; check account and model access",
        404: "not found; check the endpoint and exact model name",
        429: "rate or usage limit reached",
    }.get(status, "upstream HTTP error")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
async def health():
    endpoint = ollama_endpoint("version")
    try:
        async with ollama_client(timeout=2) as client:
            response = await client.get(endpoint)
            ollama_ok = response.is_success
            if not ollama_ok:
                status = response.status_code
                log_ollama_failure(endpoint, "health", status_hint(status), status=status)
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        log_ollama_failure(endpoint, "health", type(error).__name__)
        ollama_ok = False
    return HealthResponse(status="ok", ollama=ollama_ok)


def build_payload(req: ChatRequest, stream: bool) -> tuple[dict, dict[str, str]]:
    allowed = []
    blocked = []
    for index, message in enumerate(req.messages):
        if looks_like_prompt_injection(message.content):
            blocked.append(str(index))
        else:
            allowed.append(message)
    headers = {FILTERED_MESSAGES_HEADER: ",".join(blocked)} if blocked else {}
    if not any(message.role == "user" and message.content.strip() for message in allowed):
        raise HTTPException(422, "No usable user message remains. Send a nutrition question.", headers=headers)
    return {
        "model": req.model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(m.model_dump() for m in allowed),
        ],
        "tools": TOOLS,
        "stream": stream,
    }, headers


def add_tool_results(payload: dict, message: dict, round_number: int) -> None:
    calls = message["tool_calls"]
    if round_number >= MAX_TOOL_ROUNDS or len(calls) > MAX_CALLS_PER_ROUND:
        raise HTTPException(502, "Nutrition tool limit reached. Try a simpler request.")

    payload["messages"].append(message)
    for call in calls:
        function = call["function"]
        name = function["name"]
        payload["messages"].append({
            "role": "tool",
            "tool_name": name,
            "content": run_tool(name, function.get("arguments")),
        })


def ollama_status_error(response: httpx.Response) -> HTTPException:
    return HTTPException(response.status_code, f"Ollama rejected the request (HTTP {response.status_code}).")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response):
    payload, headers = build_payload(req, False)
    response.headers.update(headers)
    try:
        return await generate_chat(payload)
    except HTTPException as error:
        error.headers = {**(error.headers or {}), **headers}
        raise


async def generate_chat(payload: dict) -> ChatResponse:
    endpoint = ollama_endpoint("chat")
    model = payload["model"]
    try:
        async with ollama_client(timeout=None) as client:
            for round_number in range(MAX_TOOL_ROUNDS + 1):
                r = await client.post(endpoint, json=payload)
                r.raise_for_status()
                data = r.json()
                message = data["message"]
                if message["role"] != "assistant":
                    log_ollama_failure(endpoint, "chat", "unexpected upstream message role", model=model)
                    raise HTTPException(502, "Ollama returned an unexpected message role.")
                if not message.get("tool_calls"):
                    return ChatResponse(
                        model=data["model"], message=AssistantMessage(content=message["content"])
                    )
                add_tool_results(payload, message, round_number)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        log_ollama_failure(endpoint, "chat", status_hint(status), model=model, status=status)
        raise ollama_status_error(e.response) from e
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        log_ollama_failure(endpoint, "chat", type(error).__name__, model=model)
        raise HTTPException(502, "Ollama unreachable.")
    except (ValueError, KeyError, TypeError) as error:
        log_ollama_failure(endpoint, "chat", f"invalid upstream response ({type(error).__name__})", model=model)
        raise HTTPException(502, "Ollama returned an invalid response.")


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    payload, headers = build_payload(req, True)
    endpoint = ollama_endpoint("chat")
    model = payload["model"]

    async def token_stream():
        try:
            async with ollama_client(timeout=None) as client:
                for round_number in range(MAX_TOOL_ROUNDS + 1):
                    message = {"role": "assistant", "content": "", "tool_calls": []}
                    content_parts = []
                    thinking_parts = []
                    async with client.stream("POST", endpoint, json=payload) as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            chunk = json.loads(line)
                            if "error" in chunk:
                                log_ollama_failure(endpoint, "chat-stream", "upstream error event", model=model)
                                raise HTTPException(502, "Ollama could not complete the reply.")
                            delta = chunk.get("message", {})
                            if delta.get("role", "assistant") == "assistant":
                                message["tool_calls"].extend(delta.get("tool_calls") or [])
                                if thinking := delta.get("thinking"):
                                    thinking_parts.append(thinking)
                                if content := delta.get("content"):
                                    content_parts.append(content)
                                    if not message["tool_calls"]:
                                        yield content
                            if chunk.get("done"):
                                break
                        else:
                            log_ollama_failure(endpoint, "chat-stream", "upstream stream ended early", model=model)
                            raise HTTPException(502, "Ollama ended the reply early.")
                    if not message["tool_calls"]:
                        return
                    message["content"] = "".join(content_parts)
                    if thinking_parts:
                        message["thinking"] = "".join(thinking_parts)
                    add_tool_results(payload, message, round_number)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            log_ollama_failure(endpoint, "chat-stream", status_hint(status), model=model, status=status)
            yield f"[error] {ollama_status_error(e.response).detail}"
        except HTTPException as e:
            yield f"[error] {e.detail}"
        except (httpx.HTTPError, httpx.InvalidURL) as error:
            log_ollama_failure(endpoint, "chat-stream", type(error).__name__, model=model)
            yield "[error] Ollama unreachable."
        except (ValueError, KeyError, TypeError) as error:
            log_ollama_failure(endpoint, "chat-stream", f"invalid upstream response ({type(error).__name__})", model=model)
            yield "[error] Ollama returned an invalid response."

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8", headers=headers)
