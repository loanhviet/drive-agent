# Shared Drive evaluation dataset

Create a reviewed JSONL file from sanitized documents that are also present in the configured Drive folder. Do not commit private file IDs, credentials, or document text.

Each line uses this schema:

```json
{"case_id":"pdf-001","question":"...","expected_source":"Public guide.pdf","expected_locator":{"type":"page","value":2},"expected_terms":["expected phrase"],"answerable":true}
```

Use `{"type":"section","value":"Overview"}` for section-based files and `{"type":"none","value":null}` for unanswerable cases. Build at least 20 PDF cases, 5 section cases, and 5 unanswerable cases before treating quality metrics as a release gate.

Run:

```bash
.venv/bin/python -m scripts.eval_shared_drive eval/shared_drive_cases.jsonl \
  --json-output artifacts/shared-drive-eval.json \
  --markdown-output artifacts/shared-drive-eval.md
```
