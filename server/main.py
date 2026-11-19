import json
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

app = FastAPI(
    title="Ollama Chat Proxy",
    description="Proxies chat requests to a local Ollama model.",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None


class ChatResponse(BaseModel):
    model: str
    message: Message


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ollama: bool


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            ollama_ok = (await client.get(f"{OLLAMA_URL}/api/version")).is_success
    except httpx.HTTPError:
        ollama_ok = False
    return HealthResponse(status="ok", ollama=ollama_ok)


def build_payload(req: ChatRequest, stream: bool) -> dict:
    return {
        "model": req.model or DEFAULT_MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "stream": stream,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=build_payload(req, False))
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Ollama unreachable: {e}")
    data = r.json()
    return ChatResponse(model=data["model"], message=Message(**data["message"]))


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    payload = build_payload(req, True)

    # Streams plain-text tokens as they arrive from Ollama's NDJSON stream.
    async def token_stream():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        yield f"[error] {(await r.aread()).decode()}"
                        return
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            yield f"[error] {chunk['error']}"
                            return
                        if content := chunk.get("message", {}).get("content"):
                            yield content
                        if chunk.get("done"):
                            break
        except httpx.HTTPError as e:
            yield f"[error] Ollama unreachable: {e}"

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")
