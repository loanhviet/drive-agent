"""
CLI Interface - Interactive chat with the Agent.
"""

import json
import os
import sys
from agent import Agent


def print_help():
    print("""
=== Agent CLI Commands ===
  /quit     - Exit the application
  /clear    - Clear conversation history
  /audit    - Show audit log (all tool calls)
  /memory   - Show all stored memories
  /help     - Show this help message
==============================
""")


def main():
    print("\n" + "="*60)
    print("  ASSIGNMENT 1 - AI Agent with Tool Registry")
    print("  Capabilities: Google Drive | Read File | RAG Memory")
    print("="*60)

    access_token = os.getenv("APP_ACCESS_TOKEN", "")
    if not access_token:
        print("[Error: APP_ACCESS_TOKEN is required. Login through the API to obtain a JWT.]")
        return
    agent = Agent(service_api_key=access_token, session_id="cli")

    print_help()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/clear":
            agent.clear_history()
            continue

        if user_input.lower() == "/audit":
            logs = agent.get_audit_log()
            if not logs:
                print("\n[No audit logs yet]")
            else:
                print(f"\n--- AUDIT LOG ({len(logs)} entries) ---")
                for entry in logs:
                    print(json.dumps(entry, indent=2, ensure_ascii=False))
                print("--- END AUDIT LOG ---")
            continue

        if user_input.lower() == "/memory":
            try:
                from services.vectorstore import list_all_memories
                memories = list_all_memories()
                if not memories:
                    print("\n[No memories stored yet]")
                else:
                    print(f"\n--- MEMORIES ({len(memories)} entries) ---")
                    for m in memories:
                        print(f"  [{m.get('metadata', {}).get('category', 'general')}] {m['text'][:100]}")
                    print("--- END MEMORIES ---")
            except Exception as e:
                print(f"\n[Error accessing memory: {e}]")
            continue

        # Run agent
        try:
            response = agent.run(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\n[Error: {e}]")


if __name__ == "__main__":
    main()
