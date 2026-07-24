"""
FastAPI Server - Serves the Agent via HTTP API + HTML chat UI on port 9004.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from queue import Empty, Queue
import re
import traceback
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from agent import Agent
from config import GOOGLE_DRIVE_FOLDER_ID, GOOGLE_SERVICE_ACCOUNT_FILE
from services.audit import get_audit_store
from services.auth import AuthenticatedUser, AuthenticationError, get_auth_service
from services.chat import get_chat_store
from services.drive_ingestion import get_drive_ingestion_worker
from services.drive_vectorstore import get_drive_document_store
from services.ingestion import get_ingestion_store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    worker = get_drive_ingestion_worker()
    worker.start()
    try:
        yield
    finally:
        worker.stop(timeout=10.0)


app = FastAPI(title="AI Agent - Assignment 1", lifespan=lifespan)
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def resolve_request_id(raw_value: str | None) -> str:
    """Accept a safe client id or generate a server-side uuid hex."""
    if raw_value is not None:
        candidate = raw_value.strip()
        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
    return uuid.uuid4().hex


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[REQUEST_ID_HEADER],
)
app.add_middleware(RequestIdMiddleware)

# Per-session agents, isolated by authenticated actor.
sessions: dict[tuple[str, str], Agent] = {}
active_sessions: set[tuple[str, str]] = set()
active_sessions_lock = Lock()
bearer_scheme = HTTPBearer(auto_error=False)


def get_agent(session_id: str, user_id: str, access_token: str) -> Agent:
    key = (user_id, session_id)
    if key not in sessions:
        agent = Agent(service_api_key=access_token, session_id=session_id)
        history = get_chat_store().list_messages(
            user_id=user_id, session_id=session_id, limit=40
        )
        agent.conversation_history = [
            {"role": message["role"], "text": message["content"]} for message in history
        ]
        sessions[key] = agent
    else:
        sessions[key].service_api_key = access_token
    return sessions[key]


async def get_current_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> tuple[AuthenticatedUser, str]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    try:
        user = get_auth_service().verify_token(credentials.credentials)
    except (AuthenticationError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error
    return user, credentials.credentials


# ---- API Models ----

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    response: str
    tools_used: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)


class ClearRequest(BaseModel):
    session_id: str = "default"


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class SyncRequest(BaseModel):
    mode: str = Field(default="incremental", pattern="^(incremental|full)$")


class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str


def _claim_session(user_id: str, session_id: str) -> bool:
    with active_sessions_lock:
        key = (user_id, session_id)
        if key in active_sessions:
            return False
        active_sessions.add(key)
        return True


def _release_session(user_id: str, session_id: str) -> None:
    with active_sessions_lock:
        active_sessions.discard((user_id, session_id))


def _run_chat_turn(
    req: ChatRequest,
    user: AuthenticatedUser,
    token: str,
    on_status=None,
) -> tuple[str, list[str], list[dict]]:
    """Run and persist one complete turn. Failed turns are intentionally not saved."""
    session_id = req.session_id or "default"
    agent = get_agent(session_id, user.user_id, token)
    previous_history = list(getattr(agent, "conversation_history", []))
    previous_tools = list(getattr(agent, "last_tools_used", []))
    previous_citations = list(getattr(agent, "last_citations", []))
    if on_status is None:
        response = agent.run(req.message)
    else:
        response = agent.run(req.message, on_status=on_status)
    tools_used = list(agent.last_tools_used)
    citations = list(getattr(agent, "last_citations", []))
    try:
        get_chat_store().save_turn(
            user_id=user.user_id,
            session_id=session_id,
            user_message=req.message.strip(),
            assistant_message=response,
            assistant_citations=citations,
        )
    except Exception:
        if hasattr(agent, "conversation_history"):
            agent.conversation_history = previous_history
        if hasattr(agent, "last_tools_used"):
            agent.last_tools_used = previous_tools
        if hasattr(agent, "last_citations"):
            agent.last_citations = previous_citations
        raise
    return response, tools_used, citations


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---- API Endpoints ----

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    try:
        auth_service = get_auth_service()
        user = auth_service.authenticate(req.username, req.password)
        token, expires_in = auth_service.issue_token(user)
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return LoginResponse(access_token=token, expires_in=expires_in, user=user.to_dict())


@app.get("/api/auth/me")
async def get_me(auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)]):
    user, _ = auth
    return user.to_dict()

@app.post("/api/chat")
async def chat(
    req: ChatRequest,
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, token = auth
    session_id = req.session_id or "default"
    if not _claim_session(user.user_id, session_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Chat session is already processing")
    try:
        response, tools_used, citations = _run_chat_turn(req, user, token)
        return ChatResponse(response=response, tools_used=tools_used, citations=citations)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"response": f"Error: {e}", "tools_used": [], "citations": []},
        )
    finally:
        _release_session(user.user_id, session_id)


@app.post("/api/chat/stream")
async def chat_stream(
    req: ChatRequest,
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, token = auth
    session_id = req.session_id or "default"
    if not _claim_session(user.user_id, session_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Chat session is already processing")

    async def events():
        queue: Queue[tuple[str, dict]] = Queue()

        def run_worker() -> None:
            try:
                response, tools_used, citations = _run_chat_turn(
                    req, user, token, lambda payload: queue.put(("status", payload))
                )
                queue.put(
                    (
                        "final",
                        {
                            "response": response,
                            "tools_used": tools_used,
                            "citations": citations,
                        },
                    )
                )
            except Exception as error:
                queue.put(("error", {"message": f"Error: {error}"}))
            finally:
                _release_session(user.user_id, session_id)

        Thread(target=run_worker, daemon=True).start()
        try:
            while True:
                try:
                    event, payload = queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)
                    continue
                yield _sse(event, payload)
                if event in {"final", "error"}:
                    break
        finally:
            # The worker owns the session lock and may continue safely if a
            # browser disconnects before the completed turn is persisted.
            pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/clear")
async def clear_session(
    req: ClearRequest,
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, _ = auth
    session_id = req.session_id or "default"
    key = (user.user_id, session_id)
    if key in sessions:
        sessions[key].clear_history()
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/chat/history")
async def chat_history(
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
    session_id: str = "default",
):
    user, _ = auth
    messages = get_chat_store().list_messages(user_id=user.user_id, session_id=session_id)
    return {"messages": messages}


@app.get("/api/chat/sessions")
async def chat_sessions(
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, _ = auth
    return {"sessions": get_chat_store().list_sessions(user_id=user.user_id)}


@app.post("/api/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, _ = auth
    return get_chat_store().create_session(user_id=user.user_id)


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, _ = auth
    with active_sessions_lock:
        if (user.user_id, session_id) in active_sessions:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Chat session is processing")
    if not get_chat_store().delete_session(user_id=user.user_id, session_id=session_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    sessions.pop((user.user_id, session_id), None)
    _release_session(user.user_id, session_id)
    return {"status": "deleted", "session_id": session_id}


def _require_scope(user: AuthenticatedUser, scope: str) -> None:
    if scope not in user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required scope: {scope}",
        )


@app.post("/api/drive/sync", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_drive_sync(
    req: SyncRequest,
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, _ = auth
    _require_scope(user, "drive:sync")
    if not GOOGLE_DRIVE_FOLDER_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "drive_folder_not_configured"},
        )
    job = get_ingestion_store().enqueue_job(
        mode=req.mode,
        trigger="manual",
        requested_by=user.user_id,
    )
    return job


@app.get("/api/drive/sync/status")
async def drive_sync_status(
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, _ = auth
    _require_scope(user, "drive:read")
    payload = get_ingestion_store().sync_status()
    if "drive:sync" not in user.scopes:
        for key in ("active_job", "latest_job"):
            if payload.get(key):
                payload[key] = {
                    field: value
                    for field, value in payload[key].items()
                    if field not in {"requested_by", "error"}
                }
    return payload


@app.get("/api/drive/sync/jobs/{job_id}")
async def drive_sync_job(
    job_id: str,
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
):
    user, _ = auth
    _require_scope(user, "drive:sync")
    job = get_ingestion_store().get_job(job_id, include_items=True)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sync job not found")
    return job


@app.get("/api/drive/documents")
async def drive_documents(
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
    q: str = "",
    document_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    user, _ = auth
    _require_scope(user, "drive:read")
    try:
        payload = get_ingestion_store().list_documents(
            query=q,
            status=document_status,
            limit=limit,
            offset=offset,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    public_fields = {
        "file_id",
        "source_name",
        "mime_type",
        "drive_path",
        "web_view_link",
        "modified_time",
        "status",
        "total_chars",
        "page_count",
        "chunk_count",
        "indexed_at",
        "updated_at",
    }
    if "drive:sync" in user.scopes:
        public_fields.add("error")
    payload["documents"] = [
        {field: value for field, value in document.items() if field in public_fields}
        for document in payload["documents"]
    ]
    return payload


@app.get("/api/audit")
@app.get("/audit")
async def get_audit(
    auth: Annotated[tuple[AuthenticatedUser, str], Depends(get_current_auth)],
    session_id: str | None = None,
    tool: str | None = None,
    audit_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
):
    user, _ = auth
    user_filter = None if user.role == "admin" else user.user_id
    logs = get_audit_store().list_logs(
        user_id=user_filter,
        session_id=session_id,
        tool=tool,
        status=audit_status,
        limit=limit,
    )
    return {"audit_log": logs}


@app.get("/api/health")
async def health():
    return {"status": "ok", "port": 9004}


# ---- Serve HTML UI ----

@app.get("/api/ready")
async def readiness():
    checks = {
        "database": False,
        "qdrant": False,
        "drive_config": bool(GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_FILE),
        "credentials_file": bool(
            GOOGLE_SERVICE_ACCOUNT_FILE and os.path.isfile(GOOGLE_SERVICE_ACCOUNT_FILE)
        ),
        "worker": get_drive_ingestion_worker().is_alive,
    }
    try:
        get_ingestion_store().sync_status()
        checks["database"] = True
    except Exception:
        pass
    try:
        get_drive_document_store().ensure_collection()
        checks["qdrant"] = True
    except Exception:
        pass
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with STATIC_INDEX.open("r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)
