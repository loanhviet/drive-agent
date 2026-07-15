from pathlib import Path


def test_ui_has_jwt_login_and_safe_audit_rendering():
    source = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")

    assert "/api/auth/login" in source
    assert "SESSION_STORAGE_KEY" in source
    assert "Authorization': 'Bearer '" in source
    assert "/api/chat/stream" in source
    assert "/api/chat/history" in source
    assert "/api/chat/sessions" in source
    assert "createNewChat" in source
    assert "activeChatRequest" in source
    assert "sessionNavigationInFlight" in source
    assert "new AbortController()" in source
    assert "activeChatRequest || sessionNavigationInFlight" in source
    assert "if (targetSessionId !== sessionId) return;" in source
    assert "activeChatRequest !== request || sessionId !== requestSessionId" in source
    assert "if (!el || !text) return;" in source
    assert "if (typing) container.appendChild(typing);" in source
    assert "text/event-stream" in source
    assert "body.textContent = JSON.stringify(data.audit_log, null, 2);" in source
    assert "body.innerHTML = data.audit_log" not in source
    assert "Code Sandbox" not in source
