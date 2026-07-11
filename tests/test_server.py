import agent as agent_module
import pytest


@pytest.mark.anyio
async def test_health_does_not_require_external_services(client):
    response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "port": 9004}


@pytest.mark.anyio
async def test_ui_is_served_outside_project_working_directory(client, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    response = await client.get("/")

    assert response.status_code == 200
    assert "AI Agent - Assignment 1" in response.text


@pytest.mark.anyio
async def test_audit_starts_empty_without_llm_api_key(client):
    response = await client.get("/api/audit", params={"session_id": "smoke-test"})

    assert response.status_code == 200
    assert response.json() == {"audit_log": []}


@pytest.mark.anyio
async def test_clear_only_requires_session_id(client):
    response = await client.post("/api/clear", json={"session_id": "smoke-test"})

    assert response.status_code == 200
    assert response.json() == {"status": "cleared", "session_id": "smoke-test"}


@pytest.mark.anyio
async def test_chat_returns_structured_error_without_llm_api_key(client, monkeypatch):
    monkeypatch.setattr(agent_module, "ANTHROPIC_API_KEY", None)

    response = await client.post(
        "/api/chat",
        json={"session_id": "smoke-test", "message": "Hello"},
    )

    assert response.status_code == 500
    assert response.json()["tools_used"] == []
    assert "ANTHROPIC_API_KEY" in response.json()["response"]
