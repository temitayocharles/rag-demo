"""SQLite-backed MCP business records, memories and audit events (single-host POC).

Never use this store with ephemeral Vercel storage. Each caller must be authenticated by
the MCP gateway before accessing these methods. No dynamic SQL or arbitrary queries.
"""
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rag_demo.models import DocumentIn


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class McpDataStore:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS mcp_documents (
                knowledge_base TEXT NOT NULL, source_id TEXT NOT NULL,
                text TEXT NOT NULL, metadata TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (knowledge_base, source_id)
            );
            CREATE TABLE IF NOT EXISTS mcp_records (
                record_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mcp_memories (
                memory_id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                content TEXT NOT NULL, source TEXT NOT NULL,
                created_at TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS mcp_audit (
                event_id TEXT PRIMARY KEY, action TEXT NOT NULL,
                target TEXT NOT NULL, created_at TEXT NOT NULL,
                outcome TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mcp_workflow_events (
                event_id TEXT PRIMARY KEY, workflow TEXT NOT NULL,
                payload TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    def audit(self, action: str, target: str, outcome: str = "success") -> str:
        event_id = str(uuid4())
        with self.db:
            self.db.execute(
                "INSERT INTO mcp_audit VALUES (?, ?, ?, ?, ?)",
                (event_id, action, target[:256], utcnow(), outcome),
            )
        return event_id

    def save_document(self, kb: str, document: DocumentIn) -> dict:
        now = utcnow()
        with self.db:
            self.db.execute(
                """INSERT INTO mcp_documents VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(knowledge_base, source_id) DO UPDATE
                SET text=excluded.text, metadata=excluded.metadata,
                updated_at=excluded.updated_at""",
                (kb, document.source_id, document.text,
                 json.dumps(document.metadata), now),
            )
        self.audit("ingest_document", f"{kb}/{document.source_id}")
        return {"knowledge_base": kb, "source_id": document.source_id, "updated_at": now}

    def get_document(self, kb: str, source_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT text, metadata, updated_at FROM mcp_documents "
            "WHERE knowledge_base=? AND source_id=?", (kb, source_id)
        ).fetchone()
        if row is None:
            return None
        return {"knowledge_base": kb, "source_id": source_id,
                "text": row[0], "metadata": json.loads(row[1]), "updated_at": row[2]}

    def create_record(self, kind: str, payload: dict) -> dict:
        record_id = str(uuid4())
        now = utcnow()
        with self.db:
            self.db.execute(
                "INSERT INTO mcp_records VALUES (?, ?, ?, ?)",
                (record_id, kind, json.dumps(payload), now),
            )
        self.audit("create_business_record", record_id)
        return {"record_id": record_id, "kind": kind, "created_at": now}

    def save_memory(self, subject: str, content: str, source: str) -> dict:
        memory_id = str(uuid4())
        now = utcnow()
        with self.db:
            self.db.execute(
                "INSERT INTO mcp_memories VALUES (?, ?, ?, ?, ?, 0)",
                (memory_id, subject, content, source, now),
            )
        self.audit("save_memory", memory_id)
        return {"memory_id": memory_id, "subject": subject,
                "created_at": now, "verified": False}

    def queue_workflow(self, workflow: str, payload: dict) -> str:
        event_id = str(uuid4())
        with self.db:
            self.db.execute(
                "INSERT INTO mcp_workflow_events VALUES (?, ?, ?, ?, ?)",
                (event_id, workflow, json.dumps(payload), "pending", utcnow()),
            )
        return event_id

    def workflow_status(self, event_id: str, status: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE mcp_workflow_events SET status=? WHERE event_id=?",
                (status, event_id),
            )
        self.audit("trigger_workflow", event_id, status)
