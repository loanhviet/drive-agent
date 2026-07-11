# Implementation Status

Last updated: Milestone 6 is complete and awaiting review/commit.

## Completed commits

```text
60d9b78 chore: setup project
9814152 fix: setup fastapi server
dfbea8a feat: add tool registry pipeline
d93c331 feat: add file reader
169b798 feat: add google drive tools
4b9ca97 feat: add qdrant memory
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
- Last full successful checks: 76 tests passed; Ruff and compile passed.

## Current worktree changes (not committed)

Milestone 6 is implemented and ready for review:

```text
list_drive_files -> get_drive_file(file_id) -> read_file_tool(artifact_id)
```

Implemented changes:

- Added Qdrant content-hash lookup for deduplication.
- Replaced memory TODOs with fact/document save and semantic search tools.
- `document_ref` is consumed only after successful document storage.

## Required next steps

1. Review and commit Milestone 6. Proposed simple commit: `feat: add memory tools`.
2. Start Milestone 7: add Gemini/Anthropic chat providers, login UI, and agent integration tests.
3. Keep live providers opt-in; do not use or fabricate API keys or Google credentials.

## Important constraints

- Do not use or create real Google credentials for tests.
- Real Drive tests must be opt-in and skipped when credentials/folder ID are absent.
- Do not commit `.env`, `credentials.json`, `.venv`, SQLite data, Qdrant data, or temporary artifacts.
- The user prefers simple commit messages and approves a milestone before it is committed.
- No code has been pushed to GitHub.
