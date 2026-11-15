"""Register the signed Telegram webhook using local .env, without printing credentials."""
import argparse
import asyncio
from urllib.parse import urlparse

import httpx

from rag_demo.config import Settings
from rag_demo.telegram import telegram_ready, webhook_secret


async def register(base_url: str) -> None:
    settings = Settings()
    if not telegram_ready(settings):
        raise SystemExit("Set RAG_TELEGRAM_BOT_TOKEN and RAG_TELEGRAM_CHAT_ID first")
    origin = urlparse(base_url)
    if origin.scheme != "https" or not origin.netloc or origin.path not in {"", "/"}:
        raise SystemExit("Supply an HTTPS Vercel origin, without a path or query string")
    webhook = base_url.rstrip("/") + "/integrations/telegram/webhook"
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{settings.telegram_bot_token}/setWebhook",
            json={
                "url": webhook,
                "secret_token": webhook_secret(settings.telegram_bot_token or ""),
                "allowed_updates": ["message"],
                "drop_pending_updates": True,
            },
        )
        response.raise_for_status()
        if not response.json().get("ok"):
            raise SystemExit("Telegram rejected webhook registration")
    print("Telegram webhook registered successfully for the configured HTTPS origin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("origin", help="e.g. https://rag-demo-ruddy.vercel.app")
    asyncio.run(register(parser.parse_args().origin))
