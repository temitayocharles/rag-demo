"""Telegram webhook contract: secret header, allowlisted chat, no token in responses."""
import hashlib
import hmac
import logging
from typing import Any

import httpx

from rag_demo.config import Settings
from rag_demo.generator import OpenAICompatibleGenerator
from rag_demo.service import RagService

logger = logging.getLogger(__name__)
# Telegram Bot API URL paths contain tokens. Suppress outbound URL logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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


async def register_cloud_webhook(settings: Settings) -> bool:
    """Cloud-only, idempotent Telegram webhook registration using server-side secrets.

    It runs once per serverless instance startup and never prints token-bearing URLs.
    Only the production deployment may update Telegram's webhook destination.
    """
    if not telegram_ready(settings):
        return False
    webhook_url = settings.telegram_webhook_origin.rstrip("/")
    webhook_url += "/integrations/telegram/webhook"
    async with httpx.AsyncClient(timeout=20.0) as client:
        endpoint = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        if not settings.telegram_force_register:
            info = await client.post(f"{endpoint}/getWebhookInfo")
            info.raise_for_status()
            details = info.json()
            if details.get("ok") and details.get("result", {}).get("url") == webhook_url:
                logger.info("Telegram webhook already registered at expected URL")
                return True
        response = await client.post(
            f"{endpoint}/setWebhook",
            json={
                "url": webhook_url,
                "secret_token": webhook_secret(settings.telegram_bot_token or ""),
                "allowed_updates": ["message"],
                "drop_pending_updates": False,
            },
        )
        response.raise_for_status()
        if not response.json().get("ok", False):
            raise ValueError("Telegram declined webhook registration")
    logger.info("Telegram production webhook registered")
    return True


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
) -> dict[str, str] | None:
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
            elif generator is None and command != "/search":
                reply = format_brief_hit(hits[0])
            elif generator is not None and command != "/search":
                reply = await generator.answer(query, hits)
                reply += "\n\nSources: " + ", ".join(
                    f"[{i}] {hit.chunk.source_id}"
                    for i, hit in enumerate(hits, start=1)
                )
            else:
                reply = "Retrieved source excerpts (not generated answers):\n\n"
                reply += format_hits(hits)
    # Telegram can execute sendMessage from the authenticated webhook response.
    # No second outbound HTTP call is needed from the serverless function.
    return {
        "method": "sendMessage",
        "chat_id": str(settings.telegram_chat_id),
        "text": reply[:3900],
    }


def format_hits(hits: list) -> str:
    return "\n\n".join(
        f"[{i}] {hit.chunk.source_id}\n{hit.chunk.text.strip()[:850]}"
        for i, hit in enumerate(hits, start=1)
    )[:3600]


def format_brief_hit(hit: Any) -> str:
    """Return one short, attributed excerpt when LLM generation is disabled.

    /search remains available for users who explicitly request full sources.
    """
    source = " ".join(str(hit.chunk.source_id).split())[:110]
    excerpt = " ".join(str(hit.chunk.text).split())
    if len(excerpt) > 420:
        excerpt = excerpt[:417].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return (
        "Closest relevant source excerpt (not an AI-generated answer):\n\n"
        f"{excerpt}\n\nSource: {source}\n"
        "Use /search followed by your question to see more excerpts."
    )
