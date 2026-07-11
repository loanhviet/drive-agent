# Implementation Status

Last updated: Milestone 4 is complete and awaiting review/commit.

## Completed commits

```text
60d9b78 chore: setup project
9814152 fix: setup fastapi server
dfbea8a feat: add tool registry pipeline
d93c331 feat: add file reader
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
- Last full successful checks: 43 tests passed; Drive/artifact/document coverage 86%; Ruff and compile passed.

## Current worktree changes (not committed)

Milestone 4 is implemented and ready for review:

```text
list_drive_files -> get_drive_file(file_id) -> read_file_tool(artifact_id)
```

Implemented changes:

- Added `registry/context.py`: trusted tool handlers can read the authenticated actor through a `ContextVar` without changing handler schemas.
- Updated `registry/registry.py`: execution runs handlers inside this actor context.
- Added `services/artifacts.py`: short-lived user-scoped Drive download artifacts; consuming or expiring one removes its temporary file.
- Added `services/documents.py`: short-lived user-scoped cache for full extracted document content; later `save_memory(document_ref=...)` will use it in Milestone 6.
- Updated `services/file_reader.py`: `read_file(..., max_chars=None)` returns full content for the document cache while preserving the default 15,000-character preview behavior.
- Updated `services/drive_service.py`: list pagination and page-size validation.
- Updated `tools/google_drive.py`: replaces the old combined `read_drive_file` tool with `get_drive_file`.
- Updated `tools/read_file.py`: replaces `read_file` with PDF-required `read_file_tool`, consumes an artifact, extracts content, creates `document_ref`, and deletes its temporary file in `finally`.

## Required next steps

1. Review and commit Milestone 4. Proposed simple commit: `feat: add google drive tools`.
2. Start Milestone 5: configurable Gemini/OpenAI embeddings, a lightweight chunker, and persistent Qdrant vector storage.
3. Keep live Google Drive tests opt-in; do not use or fabricate credentials.

## Important constraints

- Do not use or create real Google credentials for tests.
- Real Drive tests must be opt-in and skipped when credentials/folder ID are absent.
- Do not commit `.env`, `credentials.json`, `.venv`, SQLite data, Qdrant data, or temporary artifacts.
- The user prefers simple commit messages and approves a milestone before it is committed.
- No code has been pushed to GitHub.
