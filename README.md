# RAG Demo: Local-first knowledge base POC

FastAPI ingestion and retrieval backend with explicit knowledge base selection, local SQLite
persistence, text/PDF/Word/image extraction, opt-in OCR, and configurable read-only Google Drive
and Notion sources.

**Status:** Backend proof of concept, not a conversational answer generator. The default
token-hash embedding is deterministic but **not semantic**. This is **not** production ready,
and should **not** be deployed publicly without authentication, rate limits, isolation,
persistent managed storage, logging, and security review.

## Local installation

Requirements: Python 3.11+, and [Tesseract](https://tesseract-ocr.github.io/) with `eng`
language data for scanned PDFs/images. Native-text PDFs work without OCR.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,files]'
cp .env.example .env
mkdir -p sample_docs
printf 'RAG grounds answers in trusted knowledge.\n' > sample_docs/intro.txt
python -m rag_demo.cli sync --kb local
uvicorn rag_demo.api:app --host 127.0.0.1 --port 8000
```

Check the available KBs:
```bash
curl -fsS http://127.0.0.1:8000/v1/knowledge-bases
curl -fsS -X POST http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"knowledge_base":"local","query":"trusted knowledge","top_k":3}'
```

Index one file instead:
```bash
python -m rag_demo.cli ingest ./my-file.pdf --kb local
```

Run tests and lint:
```bash
pytest -q
ruff check .
```

## Configuring knowledge sources

Edit `knowledge_bases.json` and select the KB by passing `knowledge_base` in
`POST /v1/search` and `POST /v1/documents`. The API also exposes
`GET /v1/knowledge-bases` (metadata only). An unknown KB is rejected.
KB source IDs and SQLite rows are isolated by KB. This is **not user-level authorization**.

- `local`: `{"type":"local","path":"./sample_docs"}` recursively reads supported files.
- `google_drive`: `{"type":"google_drive","folder_id":"FOLDER_ID"}` reads the
  explicitly configured immediate folder files; does not recurse subfolders.
  Create a Google service account, share that folder with its service-account address,
  and set `RAG_GOOGLE_SERVICE_ACCOUNT_FILE` in your ignored `.env`.
  Install `pip install -e '.[cloud,files]'`. Drive export supports Google Docs to text,
  and downloads supported files; Sheets/Slides export is not yet implemented.
- `notion`: `{"type":"notion","page_id":"PAGE_ID"}` reads nested text blocks from
  a Notion page. Create a Notion integration, share that page with it,
  set `RAG_NOTION_TOKEN` in `.env`. Images, file attachments, databases and
  child-page content outside the selected page are not indexed.

Use `python -m rag_demo.cli sync --kb work` only after populating real IDs and credentials.
Sync is manual and performs **read-only** source access. It reindexes returned files but
does not yet remove records when source documents are deleted. Do not schedule it without
adding deletion reconciliation, incremental sync, and retry/backoff.

Supported local inputs: UTF-8 TXT, Markdown, CSV, JSON, HTML, DOCX, PDF, PNG,
JPEG and TIFF. OCR on PDF pages with sparse text and OCR on image files rely
on Tesseract. There are per-file byte, PDF page, image pixel, and file-count limits.
Scanned tables and complex layouts may not extract accurately. Review output.

## Environment and storage

`.env.example` documents all configurable variables. `.env`, local SQLite databases,
credential JSON, and `data/` are gitignored. Do **not** commit secrets.

- `RAG_STORE_BACKEND=sqlite` (default) saves documents locally to
  `RAG_SQLITE_PATH=data/rag.sqlite3`.
- `RAG_STORE_BACKEND=memory` clears data at process restart.
- `RAG_EMBEDDING_PROVIDER=hash` needs no external service but is not semantically reliable.
- `RAG_EMBEDDING_PROVIDER=openai` requires `RAG_OPENAI_API_KEY`,
  `RAG_EMBEDDING_DIMENSIONS` and `RAG_OPENAI_EMBEDDING_MODEL`.

**Changing models or dimensions requires a full reindex**. SQLite is strictly for local,
single-machine POC. For Vercel or multiuser deployment migrate to a managed PostgreSQL
vector index (e.g. Neon pgvector), add background ingestion, auth and access control.
There is no public file-path API; file and cloud ingestion are CLI-only.

## Existing API contracts

`GET /healthz`, `POST /v1/documents`, `POST /v1/search` remain available.
Search returns chunks, scores, source IDs and metadata, not LLM-generated answers.
No frontend or chat LLM is currently included.
