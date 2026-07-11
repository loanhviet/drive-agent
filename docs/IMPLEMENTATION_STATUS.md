# Implementation Status

Last updated: Milestone 7 is complete and awaiting review/commit.

## Completed commits

```text
60d9b78 chore: setup project
9814152 fix: setup fastapi server
dfbea8a feat: add tool registry pipeline
d93c331 feat: add file reader
169b798 feat: add google drive tools
4b9ca97 feat: add qdrant memory
02983d8 feat: add memory tools
```

## Completed work

- Git repository initialized in `Assignment-1-TODO`.
- `.gitignore`, `.env.example`, baseline README, test/lint configuration, and local `.venv` are set up.
- FastAPI health/UI/clear endpoints work without external API keys.
- JWT authentication, Argon2 password hashing, SQLite users, RBAC, persistent sanitized audit logs, and the six-step registry pipeline are implemented.
- User helper: `python -m scripts.create_user <username> --role admin|user`.
- MarkItDown file reader supports TXT, Markdown, CSV, JSON, HTML, XML, PDF, DOCX, XLSX, and PPTX with validation and truncation metadata.
- Google Drive flow now follows the assignment contract: `list_drive_files` -> `get_drive_file` -> `read_file_tool`.
- Downloaded files are user-scoped temporary artifacts and are deleted after reading, after reader errors, or after expiry.
- Full extracted text is retained in a short-lived user-scoped `document_ref` cache for Milestone 6 RAG saving.
- Gemini (default) and OpenAI embedding adapters are available behind one interface; test runs inject a fake provider and do not call APIs.
- Text chunking uses 1,000 characters with 150 overlap by default.
- Qdrant supports persistent local storage for development/tests and remote server configuration for Docker deployment.
- Collection names are namespaced by embedding provider/model/dimension to prevent mixing incompatible vectors.
- `save_memory` now stores short facts as one vector and documents as chunks, with user-scoped content-hash deduplication.
- `search_memory` performs semantic search only within the authenticated user's memory and returns `insufficient_data` when no result meets the score threshold.
- Agent now supports Gemini (default) and Anthropic through a provider-neutral tool-use interface.
- The UI has JWT login, user/role display, session restoration, authenticated chat/audit calls, and safe audit rendering.
- Offline agent integration tests cover list/download/read/save/search across a new agent session.
- Last full successful checks: 87 tests passed; agent/provider/server coverage 91%; Ruff and compile passed.

## Current worktree changes (not committed)

Milestone 7 is implemented and ready for review:

```text
list_drive_files -> get_drive_file(file_id) -> read_file_tool(artifact_id)
```

Implemented changes:

- Added `services/llm.py` with Gemini, Anthropic, and scripted test providers.
- Replaced Claude-only Agent implementation with a bounded generic tool loop.
- Chat API returns tools used directly from Agent, rather than relying on audit log ordering.
- Static UI now authenticates through `/api/auth/login` and calls protected routes with a Bearer token.

## Required next steps

1. Review and commit Milestone 7. Proposed simple commit: `feat: integrate agent ui`.
2. Start Milestone 8: Docker Compose, CI, README portfolio documentation, and final demo instructions.
3. Keep live providers opt-in; do not use or fabricate API keys or Google credentials.

## Important constraints

- Do not use or create real Google credentials for tests.
- Real Drive tests must be opt-in and skipped when credentials/folder ID are absent.
- Do not commit `.env`, `credentials.json`, `.venv`, SQLite data, Qdrant data, or temporary artifacts.
- The user prefers simple commit messages and approves a milestone before it is committed.
- No code has been pushed to GitHub.
