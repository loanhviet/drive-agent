"""Secure six-step tool execution pipeline."""

import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from jsonschema import Draft202012Validator

from services.audit import AuditStore, get_audit_store, sanitize
from services.auth import AuthenticationError, get_auth_service

from .context import execution_context
from .models import ToolDefinition

PIPELINE_STEPS = (
    "validate_schema",
    "check_authentication",
    "check_scopes",
    "check_rate_limit",
    "audit_log",
    "execute_tool",
)


class SchemaValidationError(ValueError):
    pass


class ScopeError(PermissionError):
    pass


class RateLimitError(RuntimeError):
    pass


def validate_schema(tool: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate arguments using the tool's JSON Schema."""
    if not isinstance(arguments, dict):
        raise SchemaValidationError("Tool arguments must be an object")
    validator = Draft202012Validator(tool.input_schema)
    errors = sorted(validator.iter_errors(arguments), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.path)
        prefix = f"Field '{location}': " if location else ""
        raise SchemaValidationError(f"{prefix}{error.message}")
    return dict(arguments)


def check_authentication(api_key: str) -> dict[str, Any]:
    """Verify a JWT and return its active user context."""
    return get_auth_service().verify_token(api_key).to_dict()


def check_scopes(user: dict[str, Any], tool: ToolDefinition) -> bool:
    """Require every scope declared by the tool."""
    missing = sorted(set(tool.required_scopes) - set(user.get("scopes", [])))
    if missing:
        raise ScopeError(f"Missing required scopes: {', '.join(missing)}")
    return True


class RateLimiter:
    """Thread-safe sliding-window limiter keyed by actor and tool."""

    def __init__(
        self,
        max_calls: int = 20,
        window_seconds: int = 60,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_calls < 1 or window_seconds < 1:
            raise ValueError("Rate limit values must be positive")
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.clock = clock
        self._buckets: dict[str, list[float]] = {}
        self._lock = Lock()

    def check(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            active = [
                timestamp
                for timestamp in self._buckets.get(key, [])
                if now - timestamp < self.window_seconds
            ]
            if len(active) >= self.max_calls:
                retry_after = max(0.0, self.window_seconds - (now - active[0]))
                self._buckets[key] = active
                raise RateLimitError(f"Rate limit exceeded; retry in {retry_after:.1f}s")
            active.append(now)
            self._buckets[key] = active
        return True


def execute_tool(
    tool: ToolDefinition,
    arguments: dict[str, Any],
    actor: dict[str, Any] | None = None,
) -> Any:
    """Call a registered handler with validated keyword arguments."""
    if actor is None:
        return tool.handler(**arguments)
    with execution_context(actor):
        return tool.handler(**arguments)


def _new_steps() -> list[dict[str, Any]]:
    return [{"name": name, "status": "pending", "duration_ms": 0.0} for name in PIPELINE_STEPS]


def _error_code(step: str, error: Exception) -> str:
    if isinstance(error, KeyError):
        return "tool_not_found"
    if isinstance(error, SchemaValidationError):
        return "invalid_arguments"
    if isinstance(error, AuthenticationError):
        return "authentication_failed"
    if isinstance(error, ScopeError):
        return "missing_scope"
    if isinstance(error, RateLimitError):
        return "rate_limit_exceeded"
    if step == "audit_log":
        return "audit_failed"
    return "tool_execution_failed"


class ToolRegistry:
    def __init__(
        self,
        *,
        authenticator: Callable[[str], dict[str, Any]] = check_authentication,
        limiter: RateLimiter | None = None,
        audit_store: AuditStore | None = None,
    ):
        self._tools: dict[str, ToolDefinition] = {}
        self.authenticator = authenticator
        self.limiter = limiter or RateLimiter()
        self.audit_store = audit_store or get_audit_store()

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' was not found")
        return tool

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        api_key: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        audit_id = str(uuid.uuid4())
        steps = _new_steps()
        actor: dict[str, Any] = {"user_id": "anonymous", "role": "anonymous", "scopes": []}
        result: Any = None
        structured_error: dict[str, str] | None = None
        current_step = "validate_schema"
        tool: ToolDefinition | None = None
        validated_arguments: dict[str, Any] = {}

        try:
            tool, validated_arguments = self._run_step(
                steps,
                "validate_schema",
                lambda: self._validate_call(tool_name, arguments),
            )
            current_step = "check_authentication"
            actor = self._run_step(
                steps,
                current_step,
                lambda: self.authenticator(api_key),
            )
            current_step = "check_scopes"
            self._run_step(steps, current_step, lambda: check_scopes(actor, tool))
            current_step = "check_rate_limit"
            rate_key = f"{actor['user_id']}:{tool.name}"
            self._run_step(steps, current_step, lambda: self.limiter.check(rate_key))

            current_step = "audit_log"
            self._run_step(
                steps,
                current_step,
                lambda: self.audit_store.upsert(
                    self._build_entry(
                        audit_id,
                        session_id,
                        tool_name,
                        arguments,
                        actor,
                        steps,
                        "in_progress",
                        None,
                        None,
                        started,
                    )
                ),
            )

            current_step = "execute_tool"
            result = self._run_step(
                steps,
                current_step,
                lambda: execute_tool(tool, validated_arguments, actor),
            )
        except Exception as error:
            self._mark_failure_if_pending(steps, current_step, error)
            structured_error = {
                "code": _error_code(current_step, error),
                "step": current_step,
                "message": str(error),
            }
        finally:
            self._skip_pending_steps(steps, except_step="audit_log")
            audit_step = next(step for step in steps if step["name"] == "audit_log")
            if audit_step["status"] == "pending":
                self._mark_step(steps, "audit_log", "success")
            status = "failed" if structured_error else "success"
            entry = self._build_entry(
                audit_id,
                session_id,
                tool_name,
                arguments,
                actor,
                steps,
                status,
                result,
                structured_error,
                started,
            )
            try:
                self.audit_store.upsert(entry)
            except Exception as audit_error:
                self._mark_step(
                    steps,
                    "audit_log",
                    "failed",
                    error=str(audit_error),
                )
                if structured_error is None:
                    structured_error = {
                        "code": "audit_failed",
                        "step": "audit_log",
                        "message": str(audit_error),
                    }

        return {
            "ok": structured_error is None,
            "result": result if structured_error is None else None,
            "error": structured_error,
            "audit_id": audit_id,
        }

    def _validate_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[ToolDefinition, dict[str, Any]]:
        tool = self.get_tool(tool_name)
        return tool, validate_schema(tool, arguments)

    @staticmethod
    def _run_step(steps: list[dict[str, Any]], name: str, operation: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            value = operation()
        except Exception as error:
            ToolRegistry._mark_step(
                steps,
                name,
                "failed",
                error=str(error),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        ToolRegistry._mark_step(
            steps,
            name,
            "success",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return value

    @staticmethod
    def _mark_step(
        steps: list[dict[str, Any]],
        name: str,
        status: str,
        *,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        step = next(item for item in steps if item["name"] == name)
        step.update({"status": status, "duration_ms": round(duration_ms, 3)})
        if error:
            step["error"] = error

    @staticmethod
    def _mark_failure_if_pending(
        steps: list[dict[str, Any]], name: str, error: Exception
    ) -> None:
        step = next(item for item in steps if item["name"] == name)
        if step["status"] == "pending":
            ToolRegistry._mark_step(steps, name, "failed", error=str(error))

    @staticmethod
    def _skip_pending_steps(steps: list[dict[str, Any]], except_step: str) -> None:
        for step in steps:
            if step["status"] == "pending" and step["name"] != except_step:
                step["status"] = "skipped"

    @staticmethod
    def _build_entry(
        audit_id: str,
        session_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        actor: dict[str, Any],
        steps: list[dict[str, Any]],
        status: str,
        result: Any,
        error: dict[str, str] | None,
        started: float,
    ) -> dict[str, Any]:
        return {
            "audit_id": audit_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "tool": tool_name,
            "user_id": actor.get("user_id", "anonymous"),
            "role": actor.get("role", "anonymous"),
            "arguments": sanitize(arguments),
            "steps": steps,
            "status": status,
            "result": sanitize(result),
            "error": error,
            "execution_time_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def get_audit_log(self, **filters: Any) -> list[dict[str, Any]]:
        return self.audit_store.list_logs(**filters)
