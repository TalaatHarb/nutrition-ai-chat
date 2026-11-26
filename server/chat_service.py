from collections.abc import AsyncIterator

from llm_provider import LLMMessage, LLMProvider, LLMResponse, ProviderError, TextDelta
from nutrition_tools import TOOLS, run_tool

MAX_TOOL_ROUNDS = 4
MAX_CALLS_PER_ROUND = 8


class ChatService:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def add_tool_results(
        self, messages: list[LLMMessage], message: LLMMessage, round_number: int,
    ) -> None:
        if round_number >= MAX_TOOL_ROUNDS or len(message.tool_calls) > MAX_CALLS_PER_ROUND:
            raise ProviderError("Nutrition tool limit reached. Try a simpler request.")
        messages.append(message)
        for call in message.tool_calls:
            messages.append(LLMMessage(
                role="tool",
                content=run_tool(call.name, call.arguments),
                tool_name=call.name,
                tool_call_id=call.id,
            ))

    async def complete(self, messages: list[LLMMessage], model: str | None) -> LLMResponse:
        conversation = list(messages)
        for round_number in range(MAX_TOOL_ROUNDS + 1):
            response = await self.provider.complete(conversation, TOOLS, model)
            if not response.message.tool_calls:
                return response
            self.add_tool_results(conversation, response.message, round_number)
        raise ProviderError("Nutrition tool limit reached. Try a simpler request.")

    async def stream(self, messages: list[LLMMessage], model: str | None) -> AsyncIterator[str]:
        conversation = list(messages)
        for round_number in range(MAX_TOOL_ROUNDS + 1):
            response = None
            async for event in self.provider.stream(conversation, TOOLS, model):
                if isinstance(event, TextDelta):
                    yield event.content
                else:
                    response = event
            if response is None:
                raise ProviderError("The provider ended the reply without a completed turn.")
            if not response.message.tool_calls:
                return
            self.add_tool_results(conversation, response.message, round_number)
