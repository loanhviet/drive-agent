"""
Agent Core - Connects Claude LLM with the Tool Registry.
Handles the full conversation loop with tool use.
"""

import json
import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL
from registry import ToolRegistry, ToolDefinition

from tools.google_drive import ALL_DRIVE_TOOLS
from tools.read_file import ALL_READ_FILE_TOOLS
from tools.memory import ALL_MEMORY_TOOLS

ALL_TOOLS: list[ToolDefinition] = ALL_DRIVE_TOOLS + ALL_READ_FILE_TOOLS + ALL_MEMORY_TOOLS

SYSTEM_PROMPT = """\
You are a powerful AI assistant with access to the following capabilities:

1. **Google Drive**: You can list all files and read their contents (supports many formats: PDF, DOCX, XLSX, PPTX, images, etc.).
2. **Long-term Memory**: You can save and search information across conversations using semantic search (RAG).

Guidelines:
- When the user asks to read a Drive file, use list_drive_files first, then get_drive_file and read_file_tool.
- Proactively save important information to memory (user preferences, key facts, task results).
- Before answering questions about past interactions, search_memory first.
- Always respond in the same language as the user's message.
- Be concise and helpful.
"""


class Agent:
    def __init__(self, service_api_key: str, session_id: str = "default"):
        self.client: anthropic.Anthropic | None = None
        self.model = LLM_MODEL
        self.service_api_key = service_api_key
        self.session_id = session_id
        self.conversation_history: list[dict] = []

        # Initialize Tool Registry with all tools
        self.registry = ToolRegistry()
        for tool in ALL_TOOLS:
            self.registry.register(tool)

    def _get_client(self) -> anthropic.Anthropic:
        """Create the external LLM client only when chat is requested."""
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required to send chat messages. "
                "Health, UI, and audit endpoints can run without it."
            )
        if self.client is None:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self.client

    def get_tools_for_claude(self) -> list[dict]:
        return self.registry.list_tools()

    def run(self, user_message: str) -> str:
        """
        Run the full agent loop:
        1. Add user message to history
        2. Send to Claude with tools
        3. If Claude uses tools → run through Registry pipeline
        4. Loop until Claude gives final text response
        """
        print(f"\n{'#'*60}")
        print(f"  USER: {user_message}")
        print(f"{'#'*60}")

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        tools = self.get_tools_for_claude()

        while True:
            print("\n>>> Calling Claude LLM...")
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=self.conversation_history,
            )

            print(f">>> Claude stop_reason: {response.stop_reason}")

            # Final response - no more tool calls
            if response.stop_reason == "end_turn":
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })

                text_parts = [
                    block.text for block in response.content
                    if hasattr(block, "text")
                ]
                final_response = "\n".join(text_parts)

                print(f"\n{'#'*60}")
                print(f"  ASSISTANT: {final_response[:500]}")
                if len(final_response) > 500:
                    print(f"  ... [truncated, total {len(final_response)} chars]")
                print(f"{'#'*60}\n")

                return final_response

            # Process tool calls
            if response.stop_reason == "tool_use":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_use_id = block.id

                        # Run through the 6-step Registry pipeline
                        result = self.registry.call(
                            tool_name=tool_name,
                            arguments=tool_input,
                            api_key=self.service_api_key,
                            session_id=self.session_id,
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })

                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })

    def clear_history(self):
        self.conversation_history = []
        print("[Agent] Conversation history cleared.")

    def get_audit_log(self) -> list[dict]:
        return self.registry.get_audit_log()
