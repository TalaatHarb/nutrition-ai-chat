import json
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

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
SYSTEM_PROMPT = Path(__file__).with_name("nutrition_prompt.txt").read_text(encoding="utf-8").strip()
MAX_TOOL_ROUNDS = 4
MAX_CALLS_PER_ROUND = 8
FILTERED_MESSAGES_HEADER = "X-Filtered-Message-Indices"

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


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        async with ollama_client(timeout=2) as client:
            ollama_ok = (await client.get(f"{OLLAMA_URL}/api/version")).is_success
    except httpx.HTTPError:
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
    try:
        async with ollama_client(timeout=None) as client:
            for round_number in range(MAX_TOOL_ROUNDS + 1):
                r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                message = data["message"]
                if message["role"] != "assistant":
                    raise HTTPException(502, "Ollama returned an unexpected message role.")
                if not message.get("tool_calls"):
                    return ChatResponse(
                        model=data["model"], message=AssistantMessage(content=message["content"])
                    )
                add_tool_results(payload, message, round_number)
    except httpx.HTTPStatusError as e:
        raise ollama_status_error(e.response) from e
    except httpx.HTTPError:
        raise HTTPException(502, "Ollama unreachable.")
    except (ValueError, KeyError, TypeError):
        raise HTTPException(502, "Ollama returned an invalid response.")


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    payload, headers = build_payload(req, True)

    async def token_stream():
        try:
            async with ollama_client(timeout=None) as client:
                for round_number in range(MAX_TOOL_ROUNDS + 1):
                    message = {"role": "assistant", "content": "", "tool_calls": []}
                    content_parts = []
                    thinking_parts = []
                    async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            chunk = json.loads(line)
                            if "error" in chunk:
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
                            raise HTTPException(502, "Ollama ended the reply early.")
                    if not message["tool_calls"]:
                        return
                    message["content"] = "".join(content_parts)
                    if thinking_parts:
                        message["thinking"] = "".join(thinking_parts)
                    add_tool_results(payload, message, round_number)
        except httpx.HTTPStatusError as e:
            yield f"[error] {ollama_status_error(e.response).detail}"
        except HTTPException as e:
            yield f"[error] {e.detail}"
        except httpx.HTTPError:
            yield "[error] Ollama unreachable."
        except (ValueError, KeyError, TypeError):
            yield "[error] Ollama returned an invalid response."

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8", headers=headers)
