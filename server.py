"""
FastAPI Server - Serves the Agent via HTTP API + HTML chat UI on port 9004.
"""

import traceback
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from agent import Agent
from services.audit import get_audit_store
from services.auth import AuthenticatedUser, AuthenticationError, get_auth_service

app = FastAPI(title="AI Agent - Assignment 1")
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-session agents, isolated by authenticated actor.
sessions: dict[tuple[str, str], Agent] = {}
bearer_scheme = HTTPBearer(auto_error=False)


def get_agent(session_id: str, user_id: str, access_token: str) -> Agent:
    key = (user_id, session_id)
    if key not in sessions:
        sessions[key] = Agent(service_api_key=access_token, session_id=session_id)
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
    try:
        user, token = auth
        agent = get_agent(req.session_id, user.user_id, token)
        audit_before = len(agent.get_audit_log())
        response = agent.run(req.message)

        new_logs = agent.get_audit_log()[audit_before:]
        tools_used = [log["tool"] for log in new_logs]

        return ChatResponse(response=response, tools_used=tools_used)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"response": f"Error: {e}", "tools_used": []},
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

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with STATIC_INDEX.open("r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)
