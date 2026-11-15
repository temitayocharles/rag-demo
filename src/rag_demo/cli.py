"""Local-only ingestion CLI: no file-access endpoint is exposed to web clients."""
import argparse
import asyncio
from pathlib import Path

from rag_demo.api import create_app
from rag_demo.config import Settings
from rag_demo.connectors import drive_documents, local_documents, notion_document
from rag_demo.loaders import as_document


async def execute(args: argparse.Namespace) -> None:
    settings = Settings()
    app = create_app(settings)
    kb_name = args.kb or settings.default_knowledge_base
    # Ensure the configured KB exists before touching any files or external service.
    from rag_demo.knowledge import KnowledgeBases
    registry = KnowledgeBases(settings.knowledge_bases_file)
    registry.require(kb_name)
    async with app.router.lifespan_context(app):
        service = app.state.rag_service
        if args.action == "ingest":
            path = Path(args.file)
            if not path.is_file() or path.is_symlink():
                raise ValueError("source must be a regular, non-symlink file")
            if path.stat().st_size > settings.max_source_bytes:
                raise ValueError("source exceeds size limit")
            doc = as_document(f"local:{path.resolve()}", path.name, path.read_bytes(),
                              settings, {"provider": "local", "path": str(path.resolve())})
            result = await service.ingest(doc, kb_name)
            print(f"{kb_name}: indexed {doc.source_id}, {result.chunks_upserted} chunks")
        elif args.action == "sync":
            sources = registry.items[kb_name].get("sources", [])
            total = 0
            for source in sources:
                kind = source["type"]
                if kind == "local":
                    batch = local_documents(Path(source["path"]), settings)
                    for source_id, name, data, meta in batch:
                        doc = as_document(source_id, name, data, settings, meta)
                        await service.ingest(doc, kb_name)
                        total += 1
                elif kind == "google_drive":
                    folder_id = source["folder_id"]
                    if folder_id.startswith("REPLACE_"):
                        raise ValueError("configure Google Drive folder_id")
                    for source_id, name, data, meta in drive_documents(folder_id, settings):
                        doc = as_document(source_id, name, data, settings, meta)
                        await service.ingest(doc, kb_name)
                        total += 1
                elif kind == "notion":
                    page_id = source["page_id"]
                    if page_id.startswith("REPLACE_"):
                        raise ValueError("configure Notion page_id")
                    doc = notion_document(page_id, settings)
                    await service.ingest(doc, kb_name)
                    total += 1
                else:
                    raise ValueError(f"unsupported source type: {kind}")
            print(f"{kb_name}: synchronized {total} documents")
        else:
            raise ValueError("unsupported action")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG knowledge indexing")
    commands = parser.add_subparsers(dest="action", required=True)
    ingest = commands.add_parser("ingest", help="ingest one local file")
    ingest.add_argument("file")
    ingest.add_argument("--kb", help="knowledge base name")
    sync = commands.add_parser("sync", help="sync configured sources")
    sync.add_argument("--kb", help="knowledge base name")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
