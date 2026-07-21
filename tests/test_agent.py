from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from agent import Agent, SYSTEM_PROMPT
from registry.registry import ToolRegistry
from services.artifacts import ArtifactStore
from services.audit import AuditStore
from services.documents import DocumentCache
from services.embedding import set_embedding_provider_for_testing
from services.llm import LLMError, ProviderResponse, ScriptedProvider, ToolCall, create_llm_provider
from services.vectorstore import VectorStore, set_vector_store_for_testing
from tools import google_drive, read_file as read_file_module


class SemanticFakeProvider:
    name = "fake"
    model = "fake-model"
    dimension = 3

    def embed(self, texts, task_type):
        return [
            [1.0, 0.0, 0.0] if "python" in text.lower() else [0.0, 1.0, 0.0]
            for text in texts
        ]


def actor():
    return {
        "user_id": "agent-user",
        "role": "admin",
        "scopes": ["drive:read", "memory:read", "memory:write"],
    }


def make_registry(tmp_path):
    return ToolRegistry(
        authenticator=lambda _token: actor(),
        audit_store=AuditStore(str(tmp_path / "audit.db")),
    )


def test_agent_runs_tool_loop_and_returns_final_text(tmp_path):
    from registry.models import ToolDefinition

    echo_tool = ToolDefinition(
        name="echo",
        description="Echo input.",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        required_scopes=["drive:read"],
        handler=lambda message: {"echo": message},
    )
    provider = ScriptedProvider(
        [
            ProviderResponse(tool_calls=[ToolCall("call-1", "echo", {"message": "hello"})]),
            ProviderResponse(text="Tool completed."),
        ]
    )
    agent = Agent(
        "token",
        provider=provider,
        registry=make_registry(tmp_path),
        tools=[echo_tool],
    )

    statuses = []
    answer = agent.run("Please echo hello", on_status=statuses.append)

    assert answer == "Tool completed."
    assert agent.last_tools_used == ["echo"]
    assert provider.calls[1]["history"][-1]["results"][0]["result"]["result"] == {"echo": "hello"}
    assert agent.get_audit_log()[0]["tool"] == "echo"
    assert statuses == [
        {"stage": "thinking"},
        {"stage": "tool_started", "tool": "echo"},
        {"stage": "tool_finished", "tool": "echo"},
        {"stage": "thinking"},
    ]


def test_agent_stops_infinite_tool_loop(tmp_path):
    tool_call = ProviderResponse(tool_calls=[ToolCall("call", "noop", {})])
    provider = ScriptedProvider([tool_call, tool_call])
    from registry.models import ToolDefinition

    noop = ToolDefinition(
        name="noop",
        description="No operation.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        required_scopes=[],
        handler=lambda: {},
    )
    agent = Agent(
        "token",
        provider=provider,
        registry=make_registry(tmp_path),
        tools=[noop],
        max_turns=2,
    )

    with pytest.raises(RuntimeError, match="maximum of 2"):
        agent.run("loop")
    assert agent.conversation_history == []


def test_agent_directly_handles_explicit_vietnamese_drive_listing(tmp_path):
    from registry.models import ToolDefinition

    list_tool = ToolDefinition(
        name="list_drive_files",
        description="List files.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        required_scopes=["drive:read"],
        handler=lambda: {
            "total_files": 1,
            "files": [{"name": "report.pdf", "mimeType": "application/pdf"}],
        },
    )
    agent = Agent(
        "token",
        provider=ScriptedProvider([]),
        registry=make_registry(tmp_path),
        tools=[list_tool],
    )

    response = agent.run("liệt kê các file trong drive")

    assert "Đã tìm thấy 1 file" in response
    assert "report.pdf" in response
    assert agent.last_tools_used == ["list_drive_files"]


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMError, match="Unsupported LLM_PROVIDER"):
        create_llm_provider("unknown")


def test_system_prompt_searches_all_memory_for_identity_profile_questions():
    assert "not the chat session" in SYSTEM_PROMPT
    assert "identity or profile questions" in SYSTEM_PROMPT
    assert "memory_type=all" in SYSTEM_PROMPT
    assert "try another memory_type" in SYSTEM_PROMPT


def test_offline_drive_document_memory_workflow(monkeypatch, tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    set_vector_store_for_testing(VectorStore(client, "agent_memory", 3))
    set_embedding_provider_for_testing(SemanticFakeProvider())
    artifact_store = ArtifactStore()
    document_cache = DocumentCache()
    downloaded_path = tmp_path / "drive-file.txt"
    downloaded_path.write_text("Python appears in the saved Drive document.", encoding="utf-8")
    monkeypatch.setattr(google_drive, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(read_file_module, "get_artifact_store", lambda: artifact_store)
    monkeypatch.setattr(read_file_module, "get_document_cache", lambda: document_cache)
    monkeypatch.setattr(
        google_drive.drive_service,
        "list_files",
        lambda folder_id=None: [{"id": "drive-id", "name": "drive-file.txt", "mimeType": "text/plain"}],
    )
    monkeypatch.setattr(
        google_drive.drive_service,
        "download_file",
        lambda file_id: {
            "file_id": file_id,
            "file_name": "drive-file.txt",
            "mime_type": "text/plain",
            "temp_path": str(downloaded_path),
        },
    )

    class SaveWorkflowProvider:
        def __init__(self):
            self.turn = 0

        def complete(self, *, system_prompt, tools, history):
            self.turn += 1
            if self.turn == 1:
                return ProviderResponse(tool_calls=[ToolCall("list", "list_drive_files", {})])
            if self.turn == 2:
                return ProviderResponse(
                    tool_calls=[ToolCall("get", "get_drive_file", {"file_id": "drive-id"})]
                )
            if self.turn == 3:
                artifact_id = history[-1]["results"][0]["result"]["result"]["artifact_id"]
                return ProviderResponse(
                    tool_calls=[ToolCall("read", "read_file_tool", {"artifact_id": artifact_id})]
                )
            if self.turn == 4:
                document_ref = history[-1]["results"][0]["result"]["result"]["document_ref"]
                return ProviderResponse(
                    tool_calls=[ToolCall("save", "save_memory", {"document_ref": document_ref})]
                )
            return ProviderResponse(text="The file was saved to memory.")

    writer = Agent("token", provider=SaveWorkflowProvider(), registry=make_registry(tmp_path))
    assert writer.run("Read the Drive file and save it") == "The file was saved to memory."
    assert writer.last_tools_used == [
        "list_drive_files",
        "get_drive_file",
        "read_file_tool",
        "save_memory",
    ]
    assert not downloaded_path.exists()

    reader = Agent(
        "token",
        provider=ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall("search", "search_memory", {"query": "What does the document say about Python?"})]
                ),
                ProviderResponse(text="The document mentions Python."),
            ]
        ),
        registry=make_registry(tmp_path),
    )
    assert reader.run("What does the saved document say about Python?") == "The document mentions Python."
    assert reader.last_tools_used == ["search_memory"]
    assert reader.get_audit_log()[0]["tool"] == "search_memory"

    client.close()
    set_vector_store_for_testing(None)
    set_embedding_provider_for_testing(None)

def test_agent_collects_only_citations_referenced_in_final_answer(tmp_path):
    from registry.models import ToolDefinition

    search_tool = ToolDefinition(
        name="search_drive_knowledge",
        description="Search indexed Drive documents.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        required_scopes=["drive:read"],
        handler=lambda query: {
            "status": "found",
            "query": query,
            "results": [
                {
                    "citation_id": "S1",
                    "text": "Grounded evidence",
                    "citation": {
                        "id": "S1",
                        "source_name": "Guide.pdf",
                        "page_number": 2,
                    },
                },
                {
                    "citation_id": "S2",
                    "text": "Unused evidence",
                    "citation": {
                        "id": "S2",
                        "source_name": "Other.pdf",
                        "page_number": 1,
                    },
                },
            ],
        },
    )
    provider = ScriptedProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall("search", "search_drive_knowledge", {"query": "evidence"})
                ]
            ),
            ProviderResponse(text="Grounded answer [S1]."),
        ]
    )
    agent = Agent(
        "token",
        provider=provider,
        registry=make_registry(tmp_path),
        tools=[search_tool],
    )

    answer = agent.run("Find the indexed evidence")

    assert answer == "Grounded answer [S1]."
    assert [citation["source_name"] for citation in agent.last_citations] == ["Guide.pdf"]
