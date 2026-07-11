"""Per-tool-call context exposed to trusted tool handlers."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_actor_context: ContextVar[dict[str, Any] | None] = ContextVar("actor_context", default=None)


@contextmanager
def execution_context(actor: dict[str, Any]) -> Iterator[None]:
    token = _actor_context.set(dict(actor))
    try:
        yield
    finally:
        _actor_context.reset(token)


def get_current_actor() -> dict[str, Any]:
    actor = _actor_context.get()
    if actor is None:
        raise RuntimeError("Tool execution context is not available")
    return actor
