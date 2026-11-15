"""Opt-in, read-only connector adapters scoped to configured IDs."""
import io
from collections.abc import Iterator
from pathlib import Path

import httpx

from rag_demo.config import Settings
from rag_demo.loaders import SUPPORTED_SUFFIXES
from rag_demo.models import DocumentIn


def local_documents(root: Path, settings: Settings) -> Iterator[tuple[str, str, bytes, dict]]:
    if not root.is_dir():
        raise ValueError(f"local source directory not found: {root}")
    resolved = root.resolve(strict=True)
    count = 0
    for path in sorted(resolved.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if not path.resolve().is_relative_to(resolved):
            continue
        count += 1
        if count > 200:
            raise ValueError("local source file-count limit exceeded (200)")
        if path.stat().st_size > settings.max_source_bytes:
            raise ValueError(f"source too large: {path}")
        relative = path.relative_to(resolved).as_posix()
        yield (f"local:{relative}", path.name, path.read_bytes(),
               {"provider": "local", "path": relative})


def drive_documents(folder_id: str, settings: Settings) -> Iterator[tuple[str, str, bytes, dict]]:
    if not settings.google_service_account_file:
        raise ValueError("RAG_GOOGLE_SERVICE_ACCOUNT_FILE is required for Google Drive")
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        str(settings.google_service_account_file),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    api = build("drive", "v3", credentials=credentials, cache_discovery=False)
    token = None
    count = 0
    while True:
        result = api.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken,files(id,name,mimeType,size,webViewLink)",
            pageSize=100, pageToken=token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        for item in result.get("files", []):
            name, mime = item["name"], item["mimeType"]
            if mime == "application/vnd.google-apps.document":
                name += ".txt"
                request = api.files().export_media(fileId=item["id"], mimeType="text/plain")
            elif Path(name).suffix.lower() in SUPPORTED_SUFFIXES and not mime.startswith(
                "application/vnd.google-apps."
            ):
                if int(item.get("size", 0)) > settings.max_source_bytes:
                    raise ValueError(f"Drive file exceeds limit: {name}")
                request = api.files().get_media(fileId=item["id"], supportsAllDrives=True)
            else:
                continue
            count += 1
            if count > 200:
                raise ValueError("Drive file-count limit exceeded (200)")
            raw = request.execute()
            if len(raw) > settings.max_source_bytes:
                raise ValueError(f"Drive file exceeds limit: {name}")
            yield (f"drive:{item['id']}", name, raw,
                   {"provider": "google_drive", "url": item.get("webViewLink")})
        token = result.get("nextPageToken")
        if not token:
            break


def notion_document(page_id: str, settings: Settings) -> DocumentIn:
    """Read text-bearing blocks recursively. Does not download image/file attachments."""
    if not settings.notion_token:
        raise ValueError("RAG_NOTION_TOKEN is required for Notion")
    headers = {"Authorization": f"Bearer {settings.notion_token}",
               "Notion-Version": "2022-06-28"}
    lines: list[str] = []
    seen = 0

    def visit(client: httpx.Client, block_id: str, depth: int) -> None:
        nonlocal seen
        if depth > 8:
            raise ValueError("Notion nested block depth exceeds 8")
        cursor = None
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            response = client.get(f"https://api.notion.com/v1/blocks/{block_id}/children",
                                  params=params)
            response.raise_for_status()
            payload = response.json()
            for block in payload.get("results", []):
                seen += 1
                if seen > 1000:
                    raise ValueError("Notion block count exceeds 1000")
                kind = block.get("type", "")
                body = block.get(kind, {})
                pieces = body.get("rich_text", [])
                line = "".join(p.get("plain_text", "") for p in pieces)
                if line:
                    lines.append(line)
                if block.get("has_children"):
                    visit(client, block["id"], depth + 1)
            cursor = payload.get("next_cursor")
            if not payload.get("has_more") or not cursor:
                break

    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=False) as client:
        visit(client, page_id, 0)
    return DocumentIn(
        source_id=f"notion:{page_id}",
        text="\n".join(lines),
        metadata={"provider": "notion", "url": f"https://www.notion.so/{page_id.replace('-', '')}"},
    )
