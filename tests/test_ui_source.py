from pathlib import Path


def test_ui_has_jwt_login_and_safe_audit_rendering():
    source = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")

    assert "/api/auth/login" in source
    assert "SESSION_STORAGE_KEY" in source
    assert "Authorization': 'Bearer '" in source
    assert "body.textContent = JSON.stringify(data.audit_log, null, 2);" in source
    assert "body.innerHTML = data.audit_log" not in source
    assert "Code Sandbox" not in source
