# Drive Agent

Drive Agent is an AI-agent learning project built around a secure six-step tool registry, Google Drive document reading, and persistent RAG memory with Qdrant.

The repository currently contains the assignment starter code. Implementation is organized into reviewable milestones covering the API baseline, tool execution pipeline, document extraction, Google Drive, vector memory, agent integration, automated tests, and portfolio documentation.

## Planned capabilities

- Select and execute Google Drive tools from natural-language requests.
- Convert downloaded documents to Markdown with MarkItDown.
- Store user facts and chunked document knowledge in Qdrant.
- Retrieve memory after browser reloads and application restarts.
- Authenticate users with JWT and enforce role-based scopes.
- Persist sanitized six-step tool audit logs.
- Support Gemini by default with optional Anthropic and OpenAI adapters.

## Security

Never commit `.env`, Google service-account credentials, API keys, local databases, Qdrant data, or virtual environments. Copy `.env.example` to `.env` only in your local environment.

Detailed setup, architecture, testing, and demo instructions will be added as each implementation milestone is completed.
