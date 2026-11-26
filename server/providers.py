import os

from llm_provider import LLMProvider
from ollama_provider import OllamaProvider


def create_provider() -> LLMProvider:
    name = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if name == "ollama":
        return OllamaProvider.from_env()
    raise ValueError(f"Unsupported LLM_PROVIDER: {name!r}")
