"""Telegram webhook contract: secret header, allowlisted chat, no token in responses."""
import hashlib
import hmac
from typing import Any

import httpx

from rag_demo.config import Settings
from rag_demo.generator import OpenAICompatibleGenerator
from rag_demo.service import RagService


def telegram_ready(settings: Settings) -> bool:
    return bool(settings.telegram_bot_token and settings.telegram_chat_id)


def webhook_secret(bot_token: str) -> str:
    """Telegram secret_token derived from the bot token; no third secret is required."""
    return hashlib.sha256(("rag-demo:webhook:v1:" + bot_token).encode()).hexdigest()


def verify_webhook(bot_token: str | None, provided: str | None) -> bool:
    return bool(
        bot_token and provided
        and hmac.compare_digest(webhook_secret(bot_token), provided)
    )


async def send_message(bot_token: str, chat_id: str, text: str) -> None:
    """Use Telegram Bot API with no Markdown/HTML rendering of untrusted text."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3900]},
        )
        response.raise_for_status()
        if not response.json().get("ok", False):
            raise ValueError("Telegram delivery rejected")


async def process_update(
    payload: dict[str, Any],
    settings: Settings,
    service: RagService,
    generator: OpenAICompatibleGenerator | None,
) -> None:
    if not isinstance(payload, dict):
        return
    message = payload.get("message")
    if not isinstance(message, dict):
        return
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return
    if str(chat.get("id")) != str(settings.telegram_chat_id):
        # Never respond to or read content from unauthorized chats.
        return
    text = message.get("text")
    if not isinstance(text, str):
        return
    text = text.strip()[:1500]
    if not text:
        return
    command = text.split(maxsplit=1)[0].split("@")[0].lower()
    if command in {"/start", "/help"}:
        reply = (
            "Tayoca knowledge bot (public demo).\n"
            "Ask about Tayoca, its services, Sivanta, publications or community work.\n"
            "Commands: /search question, /ask question, /help.\n"
            "Without an LLM, replies are retrieved source excerpts, not AI-written answers."
        )
    else:
        query = text
        if command in {"/search", "/ask"}:
            query = text.split(maxsplit=1)[1].strip() if len(text.split(maxsplit=1)) > 1 else ""
        if not query:
            reply = "Provide a question after /search or /ask."
        else:
            hits = await service.search(query, 3, settings.default_knowledge_base)
            if not hits:
                reply = "No matching public knowledge found."
            elif command == "/ask" and generator is None:
                reply = "LLM is disabled. Here are the closest source excerpts:\n\n"
                reply += format_hits(hits)
            elif generator is not None and command != "/search":
                reply = await generator.answer(query, hits)
                reply += "\n\nSources: " + ", ".join(
                    f"[{i}] {hit.chunk.source_id}"
                    for i, hit in enumerate(hits, start=1)
                )
            else:
                reply = "Retrieved source excerpts (not generated answers):\n\n"
                reply += format_hits(hits)
    # Only the allowlisted chat ever receives a message.
    await send_message(settings.telegram_bot_token or "", str(settings.telegram_chat_id), reply)


def format_hits(hits: list) -> str:
    return "\n\n".join(
        f"[{i}] {hit.chunk.source_id}\n{hit.chunk.text.strip()[:850]}"
        for i, hit in enumerate(hits, start=1)
    )[:3600]
