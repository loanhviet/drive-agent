import pytest
from registry.models import ToolDefinition
from registry.registry import ToolRegistry
from services.audit import get_audit_store
import services.llm as llm_module
import server


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
    assert "Drive Agent" in response.text


@pytest.mark.anyio
async def test_audit_requires_authentication(client):
    response = await client.get("/api/audit")

    assert response.status_code == 401


@pytest.mark.anyio
async def test_audit_starts_empty(client, admin_token):
    response = await client.get(
        "/audit",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    assert response.json() == {"audit_log": []}


@pytest.mark.anyio
async def test_clear_only_requires_session_id(client, admin_token):
    response = await client.post(
        "/api/clear",
        json={"session_id": "smoke-test"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "cleared", "session_id": "smoke-test"}


@pytest.mark.anyio
async def test_chat_returns_structured_error_without_llm_api_key(
    client, monkeypatch, admin_token
):
    monkeypatch.setattr(llm_module, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm_module, "GEMINI_API_KEY", "")

    response = await client.post(
        "/api/chat",
        json={"session_id": "smoke-test", "message": "Hello"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 500
    assert response.json()["tools_used"] == []
    assert "GEMINI_API_KEY" in response.json()["response"]


@pytest.mark.anyio
async def test_login_and_me(client):
    login = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin-password"},
    )

    assert login.status_code == 200
    payload = login.json()
    assert payload["token_type"] == "bearer"
    assert payload["user"]["role"] == "admin"

    me = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["user_id"] == "admin-id"


@pytest.mark.anyio
async def test_login_rejects_wrong_password(client):
    response = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong-password"},
    )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_chat_sessions_can_be_created_listed_and_deleted(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    created = await client.post("/api/chat/sessions", headers=headers)

    assert created.status_code == 200
    session = created.json()
    listed = await client.get("/api/chat/sessions", headers=headers)
    assert [item["session_id"] for item in listed.json()["sessions"]] == [session["session_id"]]

    deleted = await client.delete(f"/api/chat/sessions/{session['session_id']}", headers=headers)
    assert deleted.status_code == 200
    assert (await client.get("/api/chat/sessions", headers=headers)).json() == {"sessions": []}


@pytest.mark.anyio
async def test_chat_returns_tools_used_from_agent(client, monkeypatch, admin_token):
    class FakeAgent:
        last_tools_used = ["list_drive_files", "get_drive_file"]
        last_citations = []

        def run(self, _message):
            return "I found the requested file."

    monkeypatch.setattr(server, "get_agent", lambda *_args: FakeAgent())

    response = await client.post(
        "/api/chat",
        json={"session_id": "tools-test", "message": "List files"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "I found the requested file.",
        "tools_used": ["list_drive_files", "get_drive_file"],
        "citations": [],
    }


@pytest.mark.anyio
async def test_stream_chat_persists_successful_turn_and_emits_progress(
    client, monkeypatch, admin_token
):
    class FakeAgent:
        last_tools_used = ["search_memory"]
        conversation_history = []

        def run(self, message, *, on_status=None):
            self.conversation_history.extend(
                [{"role": "user", "text": message}, {"role": "assistant", "text": "Saved answer"}]
            )
            on_status({"stage": "thinking"})
            on_status({"stage": "tool_started", "tool": "search_memory"})
            on_status({"stage": "tool_finished", "tool": "search_memory"})
            return "Saved answer"

    monkeypatch.setattr(server, "get_agent", lambda *_args: FakeAgent())
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = await client.post(
        "/api/chat/stream",
        json={"session_id": "stream-test", "message": "Find memory"},
        headers=headers,
    )

    assert response.status_code == 200
    assert "event: status" in response.text
    assert '"stage": "tool_started"' in response.text
    assert "event: final" in response.text
    history = await client.get("/api/chat/history?session_id=stream-test", headers=headers)
    assert [message["content"] for message in history.json()["messages"]] == [
        "Find memory",
        "Saved answer",
    ]


@pytest.mark.anyio
async def test_user_only_sees_own_audit_logs(
    client, isolated_services, admin_token, user_token
):
    store = get_audit_store()
    tool = ToolDefinition(
        name="visible_tool",
        description="Audit visibility test.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        required_scopes=[],
        handler=lambda: {"ok": True},
    )
    for actor in (
        {"user_id": "admin-id", "role": "admin", "scopes": []},
        {"user_id": "user-id", "role": "user", "scopes": []},
    ):
        registry = ToolRegistry(authenticator=lambda _token, actor=actor: actor, audit_store=store)
        registry.register(tool)
        registry.call("visible_tool", {}, "token")

    user_response = await client.get(
        "/audit",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    admin_response = await client.get(
        "/audit",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert {log["user_id"] for log in user_response.json()["audit_log"]} == {"user-id"}
    assert {log["user_id"] for log in admin_response.json()["audit_log"]} == {
        "admin-id",
        "user-id",
    }

@pytest.mark.anyio
async def test_drive_sync_is_admin_only_and_deduplicated(
    client,
    monkeypatch,
    tmp_path,
    admin_token,
    user_token,
):
    from services.ingestion import IngestionStore

    store = IngestionStore(str(tmp_path / "ingestion.db"))
    monkeypatch.setattr(server, "GOOGLE_DRIVE_FOLDER_ID", "folder-1")
    monkeypatch.setattr(server, "get_ingestion_store", lambda: store)

    user_response = await client.post(
        "/api/drive/sync",
        json={"mode": "incremental"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert user_response.status_code == 403

    headers = {"Authorization": f"Bearer {admin_token}"}
    first = await client.post("/api/drive/sync", json={"mode": "incremental"}, headers=headers)
    second = await client.post("/api/drive/sync", json={"mode": "full"}, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["deduplicated"] is True


@pytest.mark.anyio
async def test_drive_documents_are_visible_to_read_scope(
    client,
    monkeypatch,
    tmp_path,
    user_token,
):
    from services.ingestion import IngestionStore

    store = IngestionStore(str(tmp_path / "ingestion.db"))
    store.record_document(
        corpus_id="shared-drive",
        file_id="file-1",
        source_name="Guide.pdf",
        mime_type="application/pdf",
        drive_path="Guide.pdf",
        web_view_link="https://drive.google.com/file/d/file-1/view",
        modified_time="2026-01-01T00:00:00Z",
        source_fingerprint="fingerprint",
        status="indexed",
        last_seen_job_id="job",
        active_revision_id="revision",
    )
    monkeypatch.setattr(server, "get_ingestion_store", lambda: store)

    response = await client.get(
        "/api/drive/documents",
        headers={"Authorization": f"Bearer {user_token}"},
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["source_name"] == "Guide.pdf"
    assert "source_fingerprint" not in document
    assert "active_revision_id" not in document
