"""Provider-neutral LLM tool calling adapters for Gemini and Anthropic."""

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, LLM_MODEL, LLM_PROVIDER


class LLMError(RuntimeError):
    """Raised when a configured chat provider cannot complete a turn."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ProviderResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> ProviderResponse: ...


def sanitize_function_parameters(input_schema: dict[str, Any]) -> dict[str, Any]:
    """Keep the portable JSON Schema subset used by hosted tool APIs.

    The registry retains the original schema for strict server-side validation.
    """

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: sanitize(item)
                for key, item in value.items()
                if key not in {"additionalProperties", "anyOf"}
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    return sanitize(deepcopy(input_schema))


class AnthropicProvider:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = ANTHROPIC_API_KEY if api_key is None else api_key
        model = LLM_MODEL if model is None else model
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, *, system_prompt, tools, history) -> ProviderResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=self._messages(history),
        )
        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]
        return ProviderResponse(text=text, tool_calls=calls)

    @staticmethod
    def _messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = []
        for message in history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["text"]})
            elif message["role"] == "assistant":
                content: list[dict[str, Any]] = []
                if message.get("text"):
                    content.append({"type": "text", "text": message["text"]})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["arguments"],
                    }
                    for call in message.get("tool_calls", [])
                )
                messages.append({"role": "assistant", "content": content})
            elif message["role"] == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result["tool_call_id"],
                                "content": json.dumps(result["result"], ensure_ascii=False),
                            }
                            for result in message["results"]
                        ],
                    }
                )
        return messages


class GeminiProvider:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = GEMINI_API_KEY if api_key is None else api_key
        model = LLM_MODEL if model is None else model
        if not api_key:
            raise LLMError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(self, *, system_prompt, tools, history) -> ProviderResponse:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=sanitize_function_parameters(tool["input_schema"]),
            )
            for tool in tools
        ]
        response = self.client.models.generate_content(
            model=self.model,
            contents=self._contents(history),
            config=types.GenerateContentConfig(
                systemInstruction=system_prompt,
                tools=[types.Tool(functionDeclarations=declarations)],
            ),
        )
        parts = response.candidates[0].content.parts if response.candidates else []
        text = "\n".join(part.text for part in parts if getattr(part, "text", None))
        calls = [
            ToolCall(
                id=part.function_call.id or str(uuid.uuid4()),
                name=part.function_call.name,
                arguments=dict(part.function_call.args or {}),
            )
            for part in parts
            if getattr(part, "function_call", None)
        ]
        return ProviderResponse(text=text, tool_calls=calls)

    @staticmethod
    def _contents(history: list[dict[str, Any]]) -> list[Any]:
        from google.genai import types

        contents = []
        for message in history:
            if message["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=message["text"])]))
            elif message["role"] == "assistant":
                parts = []
                if message.get("text"):
                    parts.append(types.Part(text=message["text"]))
                parts.extend(
                    types.Part(
                        functionCall=types.FunctionCall(
                            id=call["id"], name=call["name"], args=call["arguments"]
                        )
                    )
                    for call in message.get("tool_calls", [])
                )
                contents.append(types.Content(role="model", parts=parts))
            elif message["role"] == "tool":
                parts = [
                    types.Part(
                        functionResponse=types.FunctionResponse(
                            id=result["tool_call_id"],
                            name=result["name"],
                            response={"result": result["result"]},
                        )
                    )
                    for result in message["results"]
                ]
                contents.append(types.Content(role="user", parts=parts))
        return contents


class GroqProvider:
    """Groq adapter using its OpenAI-compatible chat-completions endpoint."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = GROQ_API_KEY if api_key is None else api_key
        model = LLM_MODEL if model is None else model
        if not api_key:
            raise LLMError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model

    def complete(self, *, system_prompt, tools, history) -> ProviderResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system_prompt, history),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": sanitize_function_parameters(tool["input_schema"]),
                    },
                }
                for tool in tools
            ],
            tool_choice="auto",
        )
        if not response.choices:
            raise LLMError("Groq returned no choices")
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as error:
                raise LLMError(f"Groq returned invalid tool arguments for {call.function.name}") from error
            if not isinstance(arguments, dict):
                raise LLMError(f"Groq returned non-object tool arguments for {call.function.name}")
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        return ProviderResponse(text=message.content or "", tool_calls=calls)

    @staticmethod
    def _messages(system_prompt: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["text"]})
            elif message["role"] == "assistant":
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("text") or None,
                }
                if message.get("tool_calls"):
                    assistant_message["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                            },
                        }
                        for call in message["tool_calls"]
                    ]
                messages.append(assistant_message)
            elif message["role"] == "tool":
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": json.dumps(result["result"], ensure_ascii=False),
                    }
                    for result in message["results"]
                )
        return messages


class ScriptedProvider:
    """Offline provider used by deterministic integration tests."""

    def __init__(self, responses: list[ProviderResponse]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system_prompt, tools, history) -> ProviderResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "tools": deepcopy(tools),
                "history": deepcopy(history),
            }
        )
        if not self.responses:
            raise LLMError("Scripted provider has no response remaining")
        return self.responses.pop(0)


def create_llm_provider(provider_name: str | None = None) -> LLMProvider:
    provider_name = LLM_PROVIDER if provider_name is None else provider_name
    if provider_name == "gemini":
        return GeminiProvider()
    if provider_name == "anthropic":
        return AnthropicProvider()
    if provider_name == "groq":
        return GroqProvider()
    raise LLMError(f"Unsupported LLM_PROVIDER: {provider_name}")
