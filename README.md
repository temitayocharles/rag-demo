# RAG Demo

A modular retrieval-augmented generation **backend foundation**, with clear contracts for ingestion, chunking, embeddings and similarity search. It does **not** generate answers or call a chat LLM. Nothing is deployed.

## Architecture

`POST /v1/documents` → text normalization → deterministic overlapping chunks → embedding provider → replace-by-source vector store.

`POST /v1/search` → embed query → cosine similarity → top-k chunks with source ID, offsets and metadata.

## Local setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
ruff check .
uvicorn rag_demo.api:app --reload
```

Test calls:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS -X POST http://127.0.0.1:8000/v1/documents -H 'Content-Type: application/json' -d '{"source_id":"demo-1","text":"Retrieval augmented generation uses documents to provide grounding for model responses.","metadata":{"origin":"example"}}'
curl -fsS -X POST http://127.0.0.1:8000/v1/search -H 'Content-Type: application/json' -d '{"query":"grounding","top_k":3}'
```

Interactive OpenAPI documentation is available locally at `/docs`.

## Contracts

- **Ingestor:** plain UTF-8 text JSON only; normalize Unicode, line endings and NULs; reject blank or oversized documents. Parsing PDFs, HTML, Office files, OCR, and remote source connectors are out of scope.
- **Chunker:** character windows with overlap, stable SHA-256-derived IDs, position, source ID, character offsets, preserved metadata.
- **Embedder:** `embed(texts) -> vectors`; deterministic token-hashing embedder is a **demo stub, not a semantic model**. Select `RAG_EMBEDDING_PROVIDER=openai` and set key, model and dimensions for real embedding API calls. The OpenAI adapter assumes support for the `dimensions` input parameter.
- **Store:** `replace_source(source_id, chunks)` and `search(vector, limit)`. The in-memory implementation is non-durable, non-distributed and isolated per worker. No API authentication or tenant isolation is implemented.
- **API:** JSON input, Pydantic validation, ingestion result, similarity hits with provenance, `/healthz`.

## Configuration

See `.env.example`. Never commit `.env` or credentials. The default provider requires no API keys or network requests.

## Production readiness boundaries

This is a **production-minded structure**, not a production deployment. Before serving untrusted users: add authentication and tenant authorization; rate limits; persistent vector database such as PostgreSQL with pgvector; migrations and backup/restore; encrypted data handling and retention/deletion policies; request tracing and metrics; ingestion jobs, retry queues and dead-letter handling; document type parsing and malware checks; batch limits; injection-aware retrieval policies; explicit evaluation datasets; versioned embedding model/index migrations; deployment-specific network, timeout, logging and secret management. Define document-level ACL filtering before introducing multi-user or multi-tenant data.

Important: switching embedding provider, model, or vector dimension requires re-embedding the corpus. Source IDs are global in the demo, so there is no tenant boundary. `POST /v1/documents` replaces prior chunks for the same source ID.
