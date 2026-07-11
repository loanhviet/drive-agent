from pathlib import Path

import pytest

from registry.models import ToolDefinition
from registry.registry import RateLimiter, ToolRegistry
from services.audit import AuditStore
from services.auth import AuthenticationError


def make_tool(handler=lambda value, mode="safe": {"value": value, "mode": mode}):
    return ToolDefinition(
        name="sample_tool",
        description="A test tool.",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "mode": {"type": "string", "enum": ["safe", "fast"]},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        required_scopes=["sample:write"],
        handler=handler,
    )


def make_registry(tmp_path: Path, authenticator=None, limiter=None):
    actor = {
        "user_id": "test-user",
        "username": "tester",
        "role": "admin",
        "scopes": ["sample:write"],
    }
    registry = ToolRegistry(
        authenticator=authenticator or (lambda _token: actor),
        limiter=limiter,
        audit_store=AuditStore(str(tmp_path / "audit.db")),
    )
    registry.register(make_tool())
    return registry


def assert_failed_at(registry, result, step, code):
    assert result["ok"] is False
    assert result["error"]["step"] == step
    assert result["error"]["code"] == code
    logs = registry.get_audit_log()
    assert len(logs) == 1
    assert logs[0]["status"] == "failed"
    statuses = {item["name"]: item["status"] for item in logs[0]["steps"]}
    assert statuses[step] == "failed"
    assert statuses["audit_log"] == "success"


def test_valid_call_executes_all_six_steps(tmp_path):
    registry = make_registry(tmp_path)

    result = registry.call("sample_tool", {"value": "hello"}, "token", "session-1")

    assert result["ok"] is True
    assert result["result"] == {"value": "hello", "mode": "safe"}
    log = registry.get_audit_log()[0]
    assert log["session_id"] == "session-1"
    assert [step["status"] for step in log["steps"]] == ["success"] * 6
    assert log["execution_time_ms"] >= 0


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"value": 123},
        {"value": "ok", "mode": "invalid"},
        {"value": "ok", "unexpected": True},
    ],
)
def test_invalid_schema_is_rejected_and_audited(tmp_path, arguments):
    registry = make_registry(tmp_path)

    result = registry.call("sample_tool", arguments, "token")

    assert_failed_at(registry, result, "validate_schema", "invalid_arguments")


def test_missing_authentication_is_audited(tmp_path):
    def reject(_token):
        raise AuthenticationError("Authentication token is required")

    registry = make_registry(tmp_path, authenticator=reject)

    result = registry.call("sample_tool", {"value": "hello"}, "")

    assert_failed_at(registry, result, "check_authentication", "authentication_failed")


def test_missing_scope_is_audited(tmp_path):
    registry = make_registry(
        tmp_path,
        authenticator=lambda _token: {
            "user_id": "reader",
            "role": "user",
            "scopes": ["sample:read"],
        },
    )

    result = registry.call("sample_tool", {"value": "hello"}, "token")

    assert_failed_at(registry, result, "check_scopes", "missing_scope")


def test_rate_limit_is_enforced_and_audited(tmp_path):
    clock = [0.0]
    limiter = RateLimiter(max_calls=1, window_seconds=60, clock=lambda: clock[0])
    registry = make_registry(tmp_path, limiter=limiter)
    assert registry.call("sample_tool", {"value": "first"}, "token")["ok"] is True

    result = registry.call("sample_tool", {"value": "second"}, "token")

    assert result["error"]["code"] == "rate_limit_exceeded"
    logs = registry.get_audit_log()
    assert len(logs) == 2
    assert logs[0]["status"] == "failed"


def test_handler_exception_is_structured_and_audited(tmp_path):
    def broken_handler(value, mode="safe"):
        raise RuntimeError(f"could not process {value} in {mode}")

    registry = ToolRegistry(
        authenticator=lambda _token: {
            "user_id": "test-user",
            "role": "admin",
            "scopes": ["sample:write"],
        },
        audit_store=AuditStore(str(tmp_path / "audit.db")),
    )
    registry.register(make_tool(broken_handler))

    result = registry.call("sample_tool", {"value": "hello"}, "token")

    assert_failed_at(registry, result, "execute_tool", "tool_execution_failed")


def test_unknown_tool_is_audited(tmp_path):
    registry = make_registry(tmp_path)

    result = registry.call("unknown", {}, "token")

    assert_failed_at(registry, result, "validate_schema", "tool_not_found")


def test_sensitive_arguments_are_redacted_and_long_content_is_truncated(tmp_path):
    registry = make_registry(tmp_path)

    registry.call(
        "sample_tool",
        {"value": "x" * 600, "unexpected_token": "secret-value"},
        "token",
    )

    arguments = registry.get_audit_log()[0]["arguments"]
    assert arguments["unexpected_token"] == "[REDACTED]"
    assert "truncated" in arguments["value"]


def test_audit_persists_after_store_is_recreated(tmp_path):
    database = tmp_path / "audit.db"
    registry = ToolRegistry(
        authenticator=lambda _token: {
            "user_id": "test-user",
            "role": "admin",
            "scopes": ["sample:write"],
        },
        audit_store=AuditStore(str(database)),
    )
    registry.register(make_tool())
    registry.call("sample_tool", {"value": "persist"}, "token")

    recreated_store = AuditStore(str(database))

    assert recreated_store.list_logs()[0]["result"]["value"] == "persist"
