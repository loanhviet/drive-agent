from types import SimpleNamespace

import pytest

from services.llm import AnthropicProvider, GeminiProvider, LLMError, ToolCall


def history_with_tool_result():
    return [
        {"role": "user", "text": "Read a file"},
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [{"id": "call-1", "name": "get_drive_file", "arguments": {"file_id": "id"}}],
        },
        {
            "role": "tool",
            "results": [
                {
                    "tool_call_id": "call-1",
                    "name": "get_drive_file",
                    "result": {"ok": True, "result": {"artifact_id": "artifact"}},
                }
            ],
        },
    ]


def test_anthropic_adapter_serializes_tool_history_without_client():
    messages = AnthropicProvider._messages(history_with_tool_result())

    assert messages[0] == {"role": "user", "content": "Read a file"}
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[2]["content"][0]["tool_use_id"] == "call-1"


def test_gemini_adapter_serializes_tool_history_without_network():
    contents = GeminiProvider._contents(history_with_tool_result())

    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call.name == "get_drive_file"
    assert contents[2].parts[0].function_response.name == "get_drive_file"


def test_gemini_adapter_parses_text_and_function_call(monkeypatch):
    provider = GeminiProvider(api_key="fake-key", model="gemini-test")
    part = SimpleNamespace(
        text="I will read the file.",
        function_call=SimpleNamespace(id="call-1", name="get_drive_file", args={"file_id": "id"}),
    )
    fake_response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
    monkeypatch.setattr(provider.client.models, "generate_content", lambda **_kwargs: fake_response)

    response = provider.complete(
        system_prompt="test",
        tools=[{"name": "get_drive_file", "description": "download", "input_schema": {"type": "object"}}],
        history=[{"role": "user", "text": "Read id"}],
    )

    assert response.text == "I will read the file."
    assert response.tool_calls == [ToolCall("call-1", "get_drive_file", {"file_id": "id"})]


def test_gemini_adapter_removes_unsupported_additional_properties(monkeypatch):
    provider = GeminiProvider(api_key="fake-key", model="gemini-test")
    captured = {}
    fake_response = SimpleNamespace(candidates=[])

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return fake_response

    monkeypatch.setattr(provider.client.models, "generate_content", fake_generate_content)
    schema = {
        "type": "object",
        "properties": {
            "options": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "additionalProperties": False,
            }
        },
        "anyOf": [{"required": ["options"]}],
        "additionalProperties": False,
    }

    provider.complete(
        system_prompt="test",
        tools=[{"name": "search", "description": "search", "input_schema": schema}],
        history=[{"role": "user", "text": "Search"}],
    )

    parameters = (
        captured["config"].tools[0].function_declarations[0].parameters.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    )
    assert "additionalProperties" not in parameters
    assert "anyOf" not in parameters
    assert "additionalProperties" not in parameters["properties"]["options"]
    assert schema["additionalProperties"] is False
    assert schema["anyOf"] == [{"required": ["options"]}]


@pytest.mark.parametrize(
    ("provider", "message"),
    [(GeminiProvider, "GEMINI_API_KEY"), (AnthropicProvider, "ANTHROPIC_API_KEY")],
)
def test_provider_requires_key(provider, message):
    with pytest.raises(LLMError, match=message):
        provider(api_key="")
