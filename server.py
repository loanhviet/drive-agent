"""
FastAPI Server - Serves the Agent via HTTP API + HTML chat UI on port 9004.
"""

import json
import traceback
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import Agent

app = FastAPI(title="AI Agent - Assignment 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-session agents (keyed by session_id)
sessions: dict[str, Agent] = {}


def get_agent(session_id: str) -> Agent:
    if session_id not in sessions:
        sessions[session_id] = Agent(service_api_key="sk-admin-001")
    return sessions[session_id]


# ---- API Models ----

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    response: str
    tools_used: list[str] = []


# ---- API Endpoints ----

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        agent = get_agent(req.session_id)
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
def clear_session(req: ChatRequest):
    session_id = req.session_id or "default"
    if session_id in sessions:
        sessions[session_id].clear_history()
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/audit")
def get_audit(session_id: str = "default"):
    agent = get_agent(session_id)
    return {"audit_log": agent.get_audit_log()}


@app.get("/api/health")
def health():
    return {"status": "ok", "port": 9004}


# ---- Serve HTML UI ----

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)
