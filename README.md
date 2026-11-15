# RAG Demo: Local-first knowledge base POC

FastAPI ingestion and retrieval backend with explicit knowledge base selection, local SQLite
persistence, text/PDF/Word/image extraction, opt-in OCR, and configurable read-only Google Drive
and Notion sources.

**Status:** Local-first backend POC. Optional grounded LLM generation via `POST /v1/ask`. The default
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
- `RAG_EMBEDDING_PROVIDER=openai_compatible` selects any compatible embeddings API using
  `RAG_EMBEDDING_BASE_URL`, `RAG_EMBEDDING_API_KEY`, `RAG_EMBEDDING_MODEL`, and
  `RAG_EMBEDDING_DIMENSIONS`. Set `RAG_EMBEDDING_SEND_DIMENSIONS=false` for models
  that do not accept a `dimensions` parameter. The legacy `openai` configuration
  remains supported but is deprecated.
- `RAG_LLM_PROVIDER=openai_compatible` activates grounded generation via `POST /v1/ask`.
  Set `RAG_LLM_BASE_URL`, `RAG_LLM_API_KEY` and `RAG_LLM_MODEL`. The LLM may
  use a completely different provider from the embedding model.
- Supported protocol: non-streaming `/chat/completions` and `/embeddings` JSON APIs.
  Models with different request formats need adapters; compatible provider status must
  be checked per selected model. There is no automatic fallback or cost routing yet.
- API keys may be omitted for local OpenAI-compatible endpoints. API base URLs require
  HTTPS except for loopback HTTP; never send credentials to untrusted endpoints.

### Provider selection examples

Configure these in your ignored `.env`, replacing placeholders and leaving keys blank
only for local unauthenticated servers.

**OpenRouter LLM** (example model identifier is a placeholder):
```dotenv
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=https://openrouter.ai/api/v1
RAG_LLM_API_KEY=YOUR_OPENROUTER_KEY
RAG_LLM_MODEL=vendor/model-id
```

**NVIDIA NIM LLM** (example model identifier is a placeholder):
```dotenv
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
RAG_LLM_API_KEY=YOUR_NVIDIA_KEY
RAG_LLM_MODEL=vendor/model-id
```

**Locally hosted compatible model**:
```dotenv
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=http://127.0.0.1:11434/v1
RAG_LLM_MODEL=your-local-model
```

Only one environment block is active at a time. Switching LLMs does not require
reindexing; switching embedding models does. Provider-specific capabilities, request
parameters, model names, quotas, and pricing vary.

Example:
```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/ask \\
  -H 'Content-Type: application/json' \\
  -d '{"knowledge_base":"local","query":"What do my notes say?","top_k":3}'
```

Generated answers return retrieved source citations. Citations are provenance pointers,
not proof that each sentence is factually supported. The system does not yet provide
sentence-level citation verification or prompt-injection defenses beyond basic instructions.

**Changing embedding models or dimensions requires a full reindex**. SQLite is strictly for local,
single-machine POC. For Vercel or multiuser deployment migrate to a managed PostgreSQL
vector index (e.g. Neon pgvector), add background ingestion, auth and access control.
There is no public file-path API; file and cloud ingestion are CLI-only.

## Existing API contracts

`GET /healthz`, `POST /v1/documents`, `POST /v1/search` remain available.
Search returns chunks, scores, source IDs and metadata. With an LLM configured,\n`POST /v1/ask` returns a generated answer and source provenance. No frontend is included.


## Vercel read-only demonstration

Vercel auto-discovers `src/index.py`, which imports the existing FastAPI application.
For a **key-free public demo only**, configure Vercel environment variables:

```dotenv
RAG_DEMO_MODE=true
RAG_STORE_BACKEND=memory
RAG_LLM_PROVIDER=disabled
RAG_EMBEDDING_PROVIDER=hash
```

The demo seeds a non-sensitive sample knowledge document at each cold start.
`POST /v1/search` works against this fixture; `POST /v1/documents` returns 403,
and `POST /v1/ask` returns 503 without an LLM.
The instance-local store is ephemeral, not a shared persistent corpus.
**Do not enable private uploads or cloud connector sync on this public project.**
For real knowledge use managed persistent vectors, authentication, and a separate
background ingestion process before enabling remote writes or private content.
Local SQLite/CLI workflows remain available with `RAG_DEMO_MODE=false`.

## Bundled public Tayoca knowledge and Telegram

The repository now contains `knowledge/tayoca_public_demo_knowledge.md`, a
public-only sample package. In Vercel demo mode the default knowledge base automatically switches to `tayoca`.
When `RAG_DEMO_MODE=true`, the application indexes the Markdown document into
**ephemeral memory at cold start**. There is no Notion dependency and no private
knowledge access. The `tayoca` namespace is available through `/v1/search`.

### Telegram integration contract

Telegram uses `POST /integrations/telegram/webhook`. The endpoint is **disabled**
until both environment variables below are set:

```dotenv
RAG_TELEGRAM_BOT_TOKEN=<set-in-Vercel-secrets>
RAG_TELEGRAM_CHAT_ID=<numeric-chat-id>
```

Add these in the **Vercel project environment settings**, not in GitHub or this
public README. Redeploy production after changing runtime environment variables.
For the local registration command, set the same values in your ignored local
`.env`. Never send the bot token in a chat message or commit it to Git.

Telegram requires HTTPS and a webhook secret. To avoid needing a third secret,
the registration script derives a deterministic secret token from the bot token,
passes it to Telegram's `setWebhook`, and the server verifies it in the
`X-Telegram-Bot-Api-Secret-Token` header using a constant-time comparison.
Requests from chat IDs other than the allowlisted chat are ignored. The webhook
does not accept uploads, writes, or Telegram-origin configuration changes.

After both Vercel values are set, production is redeployed, and your local
environment is configured, run:

```bash
python scripts/register_telegram_webhook.py https://rag-demo-ruddy.vercel.app
```

The script drops any pending messages on registration to prevent replay of old
updates. Telegram chats support `/start`, `/help`, `/search <question>`,
`/ask <question>`, or a plain question. With the LLM disabled, search returns
retrieved public-source excerpts; it does **not** fabricate an LLM answer.
Only real API-backed provider configuration enables generated answers.

**Limitations:** Only one configured chat is authorized. Updates are processed
synchronously with no durable deduplication or background queue, so delivery
retries may cause duplicate messages if Telegram times out. The public demo
does not persist chat history or uploaded documents. Provider errors may
require retry. Use private chat testing first, and secure a production bot
with persistent state and rate limits before broader use.


### Cloud-only Telegram activation (no local commands)

On the deployed Vercel production app, setting `RAG_TELEGRAM_BOT_TOKEN` and
`RAG_TELEGRAM_CHAT_ID` and redeploying is sufficient. On production function
startup the app calls Telegram `setWebhook` over HTTPS using its server-side bot
credential and a derived, authenticated secret header. No Mac setup is needed.
Registration is disabled for preview and local environments. The optional
`RAG_TELEGRAM_WEBHOOK_ORIGIN` changes the production callback origin (default:
`https://rag-demo-ruddy.vercel.app`), and `RAG_TELEGRAM_AUTO_REGISTER=false`
disables automatic registration. Only production can auto-register; otherwise
use the existing manual registration script.

Startup re-registers the same destination per cold instance. Telegram treats
this operation as a configuration update; frequent cold starts should be
monitored. A failed registration logs only the exception class without URLs,
so credentials cannot appear in application logs. This remains a demo without
persistent Telegram update deduplication or guaranteed processing.


### Telegram serverless lifecycle follow-up

Registration occurs as a FastAPI response background task after the first
production `GET /` request to avoid blocking FastAPI startup. Production-only
registration uses the bot token already held in Vercel and the same fixed HTTPS
webhook URL. On each new serverless instance, the first root request can repeat
this idempotent registration. Trigger `GET /` once after a deployment and review
Vercel runtime logs for registration success before treating the bot as operational.

For webhook replies, the API now responds with Telegram's supported
`sendMessage` JSON payload directly to authenticated Telegram requests rather
than making an outbound sendMessage HTTP call. This avoids the extra network
round trip. Unauthorized chats receive no reply.


### Telegram token logging and rotation

Telegram bot API URLs contain the bot token. The application suppresses httpx and
httpcore INFO-level request URL logging. Existing runtime log entries cannot be
retroactively redacted by code changes; rotate a bot token if it appeared there.

Registration now checks Telegram getWebhookInfo first to avoid repeated setWebhook
and 429 responses. When **rotating the token**, set the additional production
Vercel variable `RAG_TELEGRAM_FORCE_REGISTER=true` for one redeploy to update the
webhook's derived secret even when the URL is unchanged. After verifying success,
set it back to false or remove it, then redeploy. Keep tokens out of chat and Git.

## MCP server for agents and n8n

The application now exposes a **Streamable HTTP** Model Context Protocol server at
`/mcp`. It reuses the same FastAPI retrieval engine. Both protocol generations
supported by the official Python MCP SDK are handled in stateless JSON-response
mode. Clients such as n8n should use **HTTP Streamable** and **Bearer** credentials.

MCP is **disabled by default** and responds HTTP 503 until an MCP token is
configured. Set `RAG_MCP_READ_TOKEN` (32+ random characters) in the hosting
environment to allow read-only clients. Set a **different**
`RAG_MCP_WRITE_TOKEN` to authorize writes on a durable host. Never paste
tokens in conversation or commit them to Git.

Example client connection:

```text
Endpoint: https://rag-demo-ruddy.vercel.app/mcp
Transport: Streamable HTTP
Authentication: Bearer
Token: <configured RAG_MCP_READ_TOKEN>
```

### MCP tools

| Tool | Permission | Notes |
|---|---|---|
| `list_knowledge_bases` | read | Public demo lists only `tayoca` |
| `search_knowledge` | read | Relevant chunks with scores and provenance |
| `ask_knowledge` | read | Returns disabled when no LLM is configured |
| `get_document` | read | Bundled public or authorized registered document |
| `get_source_metadata` | read | Provenance for known source IDs |
| `ingest_document` | write | Index and register a text document |
| `create_business_record` | write | Immutable structured record, <= 16 KiB |
| `save_memory` | write | Unverified memory with provenance |
| `trigger_workflow` | write | Signed event to a configured n8n webhook only |

**Write safety:** All writes are rejected with HTTP 403 on the Vercel public
demo, even with a valid write token. Vercel uses in-memory storage and must
never acknowledge a business write that cannot persist. A separate durable
SQLite single-host deployment supports writes; for customer-grade cloud
storage, implement an authenticated tenant-scoped PostgreSQL adapter first.
The local SQLite MCP record, memory and audit tables are for single-tenant
proof-of-concept usage only.

Workflow execution is disabled unless `RAG_MCP_WORKFLOW_URL` and a
32+ character `RAG_MCP_WORKFLOW_SECRET` are configured on the durable host.
`trigger_workflow` accepts only `RAG_MCP_WORKFLOW_NAME` (default `default`)
and sends HMAC-SHA256 in `X-RAG-Signature`. The receiver MUST validate this
signature before acting and deduplicate `X-RAG-Event-ID`. The return value
`delivered` means HTTP accepted, **not** that the workflow completed.

Before connecting production agents, add stronger multi-tenant identity
authorization, user-scoped audit, quotas, transactional outbox with replay
and recovery, encryption, and retention/deletion policies. A shared token
is a single-client bootstrap, not an OAuth deployment.
