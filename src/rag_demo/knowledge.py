import json
import re
from pathlib import Path
from typing import Any


class KnowledgeBases:
    """Explicit allowlist of searchable knowledge bases and their sources."""

    def __init__(self, path: Path):
        self.path = path
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.items: dict[str, dict[str, Any]] = payload["knowledge_bases"]
        if not self.items:
            raise ValueError("at least one knowledge base is required")
        for name, definition in self.items.items():
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
                raise ValueError(f"invalid knowledge base name: {name}")
            if not isinstance(definition.get("sources", []), list):
                raise ValueError(f"invalid sources for {name}")

    def require(self, name: str) -> str:
        if name not in self.items:
            raise ValueError(f"unknown knowledge base: {name}")
        return name

    def summary(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": item.get("description", ""),
             "source_types": [source["type"] for source in item.get("sources", [])]}
            for name, item in self.items.items()
        ]
