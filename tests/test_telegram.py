import asyncio

from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings
from rag_demo.telegram import process_update, telegram_ready, verify_webhook, webhook_secret


def test_secret_and_configuration():
    settings = Settings(_env_file=None, telegram_bot_token="123:example", telegram_chat_id="123")
    assert telegram_ready(settings)
    assert verify_webhook("123:example", webhook_secret("123:example"))
    assert not verify_webhook("123:example", webhook_secret("different"))
    assert not verify_webhook("123:example", None)


def test_webhook_is_disabled_until_configured():
    with TestClient(create_app(Settings(_env_file=None, store_backend="memory"))) as client:
        assert client.post("/integrations/telegram/webhook", json={}).status_code == 503


def test_invalid_secret_rejected_without_sending():
    settings = Settings(
        _env_file=None, store_backend="memory", telegram_bot_token="123:example",
        telegram_chat_id="789", default_knowledge_base="tayoca"
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/integrations/telegram/webhook", json={"message": {
            "chat": {"id": 789}, "text": "/start"
        }}).status_code == 403


def test_only_allowed_chat_gets_reply(monkeypatch):
    sent = []

    async def capture(bot_token, chat_id, content):
        sent.append((bot_token, chat_id, content))

    monkeypatch.setattr("rag_demo.telegram.send_message", capture)
    settings = Settings(
        _env_file=None, store_backend="memory", telegram_bot_token="123:example",
        telegram_chat_id="789", default_knowledge_base="tayoca"
    )
    with TestClient(create_app(settings)) as client:
        for chat in [10, 789]:
            result = client.post(
                "/integrations/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:example")},
                json={"update_id": 1, "message": {"chat": {"id": chat}, "text": "/help"}},
            )
            assert result.status_code == 200
    assert len(sent) == 1 and sent[0][1] == "789"
    assert "Tayoca knowledge bot" in sent[0][2]


def test_demo_uses_public_knowledge():
    settings = Settings(
        _env_file=None, store_backend="memory", demo_mode=True,
        default_knowledge_base="tayoca",
    )
    with TestClient(create_app(settings)) as client:
        hits = client.post("/v1/search", json={
            "knowledge_base": "tayoca", "query": "Who founded Tayoca?"
        })
        assert hits.status_code == 200
        assert len(hits.json()["hits"]) > 0
        all_text = " ".join(hit["chunk"]["text"] for hit in hits.json()["hits"])
        assert "Tayoca" in all_text
