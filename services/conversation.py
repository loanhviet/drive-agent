"""Build a bounded, structurally valid conversation window for hosted LLMs."""

import json
from typing import Any


def _message_size(message: dict[str, Any]) -> int:
    return len(json.dumps(message, ensure_ascii=False, default=str))


def _group_turns(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group one user request with every assistant/tool message it produced."""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in history:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def select_recent_history(
    history: list[dict[str, Any]],
    max_chars: int,
    *,
    minimum_turns: int = 2,
) -> list[dict[str, Any]]:
    """Return recent complete turns within a character budget.

    Whole turns are retained so a tool result is never separated from its
    preceding assistant tool call. The newest turns win. ``minimum_turns`` is
    best-effort: it may exceed the budget only when those turns are themselves
    larger than the configured window.
    """
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if not isinstance(minimum_turns, int) or isinstance(minimum_turns, bool) or minimum_turns < 1:
        raise ValueError("minimum_turns must be a positive integer")
    if not history:
        return []

    selected: list[list[dict[str, Any]]] = []
    selected_size = 0
    for turn in reversed(_group_turns(history)):
        turn_size = sum(_message_size(message) for message in turn)
        if len(selected) >= minimum_turns and selected_size + turn_size > max_chars:
            break
        selected.append(turn)
        selected_size += turn_size

    return [message for turn in reversed(selected) for message in turn]
