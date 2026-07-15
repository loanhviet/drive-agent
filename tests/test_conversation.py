import pytest

from services.conversation import select_recent_history


def test_context_keeps_complete_recent_tool_turns():
    history = [
        {"role": "user", "text": "old question"},
        {"role": "assistant", "text": "old answer"},
        {"role": "user", "text": "read the file"},
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [{"id": "call-1", "name": "read_file_tool", "arguments": {}}],
        },
        {
            "role": "tool",
            "results": [{"tool_call_id": "call-1", "result": {"content": "preview"}}],
        },
        {"role": "assistant", "text": "file read"},
        {"role": "user", "text": "save it"},
    ]

    selected = select_recent_history(history, max_chars=300, minimum_turns=2)

    assert selected[0]["text"] == "read the file"
    assert [message["role"] for message in selected] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]


def test_context_prefers_newest_turns_with_best_effort_minimum():
    history = [
        {"role": "user", "text": "a" * 100},
        {"role": "assistant", "text": "b" * 100},
        {"role": "user", "text": "latest"},
    ]

    selected = select_recent_history(history, max_chars=20, minimum_turns=1)

    assert selected == [{"role": "user", "text": "latest"}]


@pytest.mark.parametrize(("max_chars", "minimum_turns"), [(0, 1), (10, 0), (True, 1)])
def test_context_validates_limits(max_chars, minimum_turns):
    with pytest.raises(ValueError):
        select_recent_history([], max_chars=max_chars, minimum_turns=minimum_turns)
