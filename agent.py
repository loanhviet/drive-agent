"""Agent core: provider-neutral tool-use loop connected to ToolRegistry."""

from typing import Any, Callable
import re
import unicodedata

from config import AGENT_CONTEXT_MAX_CHARS, MAX_AGENT_TURNS
from registry import ToolDefinition, ToolRegistry
from services.conversation import select_recent_history
from services.llm import LLMProvider, create_llm_provider
from tools.drive_knowledge import ALL_DRIVE_KNOWLEDGE_TOOLS
from tools.google_drive import ALL_DRIVE_TOOLS
from tools.memory import ALL_MEMORY_TOOLS
from tools.read_file import ALL_READ_FILE_TOOLS

ALL_TOOLS: list[ToolDefinition] = (
    ALL_DRIVE_TOOLS + ALL_READ_FILE_TOOLS + ALL_MEMORY_TOOLS + ALL_DRIVE_KNOWLEDGE_TOOLS
)

SYSTEM_PROMPT = """\
You are a helpful AI assistant with Google Drive and long-term memory tools.

Tool rules:
- To list Drive files, use list_drive_files.
- To find a Drive file by its name, use search_drive_files. Use search_drive_knowledge only when
  searching inside the content of documents already indexed from the shared Drive folder.
- To read a Drive file directly, use list_drive_files if needed, then get_drive_file, then read_file_tool.
- For questions about documents already indexed from the shared Drive folder, use
  search_drive_knowledge instead of downloading or saving the file again.
- Claims from search_drive_knowledge must use its exact citation IDs such as [S1]. If returned
  chunks do not directly answer the question, say the indexed Drive corpus has insufficient data.
- read_file_tool may return only a preview when is_truncated is true. Use read_document_segment
  with its document_ref and next_offset to read any later content needed for the answer. Continue
  with each returned next_offset until enough evidence is found or next_offset is null; never claim
  a partial read is the complete document.
- When the user explicitly asks you to remember a fact, use save_memory with content.
- When the user asks to save a file that was just read, use save_memory with document_ref from read_file_tool.
- Long-term memory is scoped to the authenticated user, not the chat session. Starting a new chat
  does not clear saved memory.
- When the user asks what is remembered or which documents are saved, use list_saved_memories.
  Use search_memory instead when the user asks a semantic question about saved information.
- Before answering explicit saved preferences, use search_memory with memory_type=fact.
- For identity or profile questions such as name, education, experience, or skills, use
  search_memory with memory_type=all because the answer may be a fact or in a saved profile/CV.
- Before answering questions about saved documents, use search_memory with memory_type=document.
- A found search result is only semantically related. If its text does not explicitly answer the
  question, refine the query or try another memory_type instead of treating it as the answer.
- Pass source_name only when the user identifies a saved source; a distinctive partial name is enough.
- When search_memory returns document citations, cite the source name and section or chunk index.
- Treat returned memory text as the sole evidence for saved-document answers. Ignore related
  pretraining knowledge and do not infer properties that the returned text does not explicitly state.
  Prefer an incomplete grounded answer over adding an unsupported claim.
- Cite saved documents using source_name plus section or chunk index. Never invent a URL from file_id.
- If search_memory reports insufficient_data, say you do not have enough saved information; never invent an answer.
- Always respond in the same language as the user and be concise.
"""


class Agent:
    def __init__(
        self,
        service_api_key: str,
        session_id: str = "default",
        *,
        provider: LLMProvider | None = None,
        registry: ToolRegistry | None = None,
        tools: list[ToolDefinition] | None = None,
        max_turns: int = MAX_AGENT_TURNS,
    ):
        self._provider = provider
        self.model = None
        self.service_api_key = service_api_key
        self.session_id = session_id
        self.max_turns = max_turns
        self.conversation_history: list[dict[str, Any]] = []
        self.last_tools_used: list[str] = []
        self.last_citations: list[dict[str, Any]] = []
        self.registry = registry or ToolRegistry()
        for tool in tools or ALL_TOOLS:
            self.registry.register(tool)

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_llm_provider()
        return self._provider

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return self.registry.list_tools()

    def run(
        self,
        user_message: str,
        *,
        on_status: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Run the user/LLM/tool loop until a text response is produced."""
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty")
        starting_history_length = len(self.conversation_history)
        previous_tools_used = self.last_tools_used
        previous_citations = self.last_citations
        self.conversation_history.append({"role": "user", "text": user_message.strip()})
        self.last_tools_used = []
        self.last_citations = []
        available_citations: dict[str, dict[str, Any]] = {}
        tools = self.get_tools_for_llm()

        try:
            if self._is_drive_listing_request(user_message):
                return self._list_drive_files_directly(on_status)
            self._emit_status(on_status, "thinking")
            for _ in range(self.max_turns):
                response = self.provider.complete(
                    system_prompt=SYSTEM_PROMPT,
                    tools=tools,
                    history=select_recent_history(
                        self.conversation_history,
                        AGENT_CONTEXT_MAX_CHARS,
                    ),
                )
                if not response.tool_calls:
                    self.last_citations = self._citations_for_response(
                        response.text,
                        available_citations,
                    )
                    self.conversation_history.append({"role": "assistant", "text": response.text})
                    return response.text

                calls = [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ]
                self.conversation_history.append(
                    {"role": "assistant", "text": response.text, "tool_calls": calls}
                )
                tool_results = []
                for call in response.tool_calls:
                    self._emit_status(on_status, "tool_started", tool=call.name)
                    result = self.registry.call(
                        tool_name=call.name,
                        arguments=call.arguments,
                        api_key=self.service_api_key,
                        session_id=self.session_id,
                    )
                    self.last_tools_used.append(call.name)
                    if call.name == "search_drive_knowledge" and result["ok"]:
                        self._capture_drive_citations(result["result"], available_citations)
                    tool_results.append(
                        {"tool_call_id": call.id, "name": call.name, "result": result}
                    )
                    self._emit_status(
                        on_status,
                        "tool_finished" if result["ok"] else "tool_failed",
                        tool=call.name,
                    )
                self.conversation_history.append({"role": "tool", "results": tool_results})
                self._emit_status(on_status, "thinking")

            raise RuntimeError(f"Agent exceeded the maximum of {self.max_turns} tool-use turns")
        except Exception:
            self.conversation_history = self.conversation_history[:starting_history_length]
            self.last_tools_used = previous_tools_used
            self.last_citations = previous_citations
            raise

    @staticmethod
    def _capture_drive_citations(
        payload: dict[str, Any],
        available: dict[str, dict[str, Any]],
    ) -> None:
        for result in payload.get("results", []):
            citation = result.get("citation") or {}
            citation_id = f"S{len(available) + 1}"
            result["citation_id"] = citation_id
            citation["id"] = citation_id
            result["citation"] = citation
            available[citation_id] = citation

    @staticmethod
    def _citations_for_response(
        response: str,
        available: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ordered_ids = dict.fromkeys(re.findall(r"\[(S\d+)\]", response))
        return [available[citation_id] for citation_id in ordered_ids if citation_id in available]

    @staticmethod
    def _emit_status(
        callback: Callable[[dict[str, Any]], None] | None,
        stage: str,
        *,
        tool: str | None = None,
    ) -> None:
        if callback is not None:
            callback({"stage": stage, **({"tool": tool} if tool else {})})

    def _list_drive_files_directly(
        self, on_status: Callable[[dict[str, Any]], None] | None
    ) -> str:
        """Make explicit Drive-list requests reliable across LLM providers."""
        tool_name = "list_drive_files"
        self._emit_status(on_status, "tool_started", tool=tool_name)
        result = self.registry.call(
            tool_name=tool_name,
            arguments={},
            api_key=self.service_api_key,
            session_id=self.session_id,
        )
        self.last_tools_used.append(tool_name)
        self._emit_status(
            on_status,
            "tool_finished" if result["ok"] else "tool_failed",
            tool=tool_name,
        )
        if not result["ok"]:
            raise RuntimeError(result["error"]["message"])

        payload = result["result"]
        files = payload["files"]
        if not files:
            response = "Không tìm thấy file nào trong Google Drive."
        else:
            lines = [f"Đã tìm thấy {payload['total_files']} file trong Google Drive:"]
            lines.extend(
                f"- {file['name']} ({file['mimeType'] or 'unknown type'})" for file in files
            )
            response = "\n".join(lines)
        self.conversation_history.append({"role": "assistant", "text": response})
        return response

    @staticmethod
    def _is_drive_listing_request(message: str) -> bool:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFD", message.casefold())
            if unicodedata.category(character) != "Mn"
        )
        asks_to_list = any(phrase in normalized for phrase in ("liet ke", "danh sach", "list"))
        mentions_drive = any(term in normalized for term in ("drive", "file", "tep", "tai lieu"))
        return asks_to_list and mentions_drive

    def clear_history(self) -> None:
        self.conversation_history = []
        self.last_tools_used = []
        self.last_citations = []

    def get_audit_log(self) -> list[dict[str, Any]]:
        return self.registry.get_audit_log(session_id=self.session_id)
