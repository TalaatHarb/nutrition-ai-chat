import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chat_service import ChatService
from llm_provider import LLMMessage, LLMProvider, ProviderError
from prompt_filter import looks_like_prompt_injection
from providers import create_provider

SYSTEM_PROMPT = Path(__file__).with_name("nutrition_prompt.txt").read_text(encoding="utf-8").strip()
FILTERED_MESSAGES_HEADER = "X-Filtered-Message-Indices"

app = FastAPI(
    title="Nutrition Assistant",
    description="Nutrition chat powered by a configurable LLM provider.",
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

provider = create_provider()


def get_provider() -> LLMProvider:
    return provider


ProviderDependency = Annotated[LLMProvider, Depends(get_provider)]


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


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
async def health(llm: ProviderDependency):
    return HealthResponse(status="ok", provider_available=await llm.health())


def prepare_messages(req: ChatRequest) -> tuple[list[LLMMessage], dict[str, str]]:
    allowed = []
    blocked = []
    for index, message in enumerate(req.messages):
        if looks_like_prompt_injection(message.content):
            blocked.append(str(index))
        else:
            allowed.append(LLMMessage(role=message.role, content=message.content))
    headers = {FILTERED_MESSAGES_HEADER: ",".join(blocked)} if blocked else {}
    if not any(message.role == "user" and message.content.strip() for message in allowed):
        raise HTTPException(422, "No usable user message remains. Send a nutrition question.", headers=headers)
    return [LLMMessage(role="system", content=SYSTEM_PROMPT), *allowed], headers


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response, llm: ProviderDependency):
    messages, headers = prepare_messages(req)
    response.headers.update(headers)
    try:
        result = await ChatService(llm).complete(messages, req.model)
    except ProviderError as error:
        raise HTTPException(error.status_code, error.detail, headers=headers) from error
    return ChatResponse(model=result.model, message=AssistantMessage(content=result.message.content))


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest, llm: ProviderDependency):
    messages, headers = prepare_messages(req)

    async def token_stream():
        try:
            async for content in ChatService(llm).stream(messages, req.model):
                yield content
        except ProviderError as error:
            yield f"[error] {error.detail}"

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8", headers=headers)
