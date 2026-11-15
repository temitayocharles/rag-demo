from fastapi.testclient import TestClient

from rag_demo.api import create_app
from rag_demo.config import Settings
from rag_demo.telegram import telegram_ready, verify_webhook, webhook_secret


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


def test_only_allowed_chat_gets_reply():
    settings = Settings(
        _env_file=None, store_backend="memory", telegram_bot_token="123:example",
        telegram_chat_id="789", default_knowledge_base="tayoca"
    )
    with TestClient(create_app(settings)) as client:
        unauthorized = client.post(
            "/integrations/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:example")},
            json={"update_id": 1, "message": {"chat": {"id": 10}, "text": "/help"}},
        )
        assert unauthorized.status_code == 200
        assert unauthorized.json() == {"ok": True}
        allowed = client.post(
            "/integrations/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:example")},
            json={"update_id": 2, "message": {"chat": {"id": 789}, "text": "/help"}},
        )
        assert allowed.status_code == 200
        assert allowed.json()["method"] == "sendMessage"
        assert allowed.json()["chat_id"] == "789"
        assert "Tayoca knowledge bot" in allowed.json()["text"]


def test_demo_uses_public_knowledge():
    settings = Settings(
        _env_file=None, store_backend="memory", demo_mode=True,
    )
    with TestClient(create_app(settings)) as client:
        hits = client.post("/v1/search", json={
            "knowledge_base": "tayoca", "query": "Who founded Tayoca?"
        })
        assert hits.status_code == 200
        assert len(hits.json()["hits"]) > 0
        all_text = " ".join(hit["chunk"]["text"] for hit in hits.json()["hits"])
        assert "Tayoca" in all_text


def test_cloud_webhook_registration_uses_token_without_exposing_it(monkeypatch):
    import asyncio

    import httpx

    from rag_demo.telegram import register_cloud_webhook

    sent = {}

    def respond(request: httpx.Request) -> httpx.Response:
        sent["url"] = str(request.url)
        sent["payload"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"ok": True, "result": True})

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr("rag_demo.telegram.httpx.AsyncClient", client_factory)
    settings = Settings(
        _env_file=None, telegram_bot_token="123:fake", telegram_chat_id="789"
    )
    assert asyncio.run(register_cloud_webhook(settings))
    assert sent["url"] == "https://api.telegram.org/bot123:fake/setWebhook"
    assert sent["payload"]["url"] == (
        "https://rag-demo-ruddy.vercel.app/integrations/telegram/webhook"
    )
    assert sent["payload"]["secret_token"] == webhook_secret("123:fake")
    assert sent["payload"]["drop_pending_updates"] is False


def test_cloud_auto_register_runs_only_in_production(monkeypatch):
    from fastapi.testclient import TestClient

    calls = []

    async def capture(_settings):
        calls.append("registered")
        return True

    monkeypatch.setattr("rag_demo.api.register_cloud_webhook", capture)
    settings = Settings(
        _env_file=None, store_backend="memory",
        telegram_bot_token="123:fake", telegram_chat_id="789",
    )
    monkeypatch.setenv("VERCEL_ENV", "preview")
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").status_code == 200
    assert not calls
    monkeypatch.setenv("VERCEL_ENV", "production")
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").status_code == 200
        assert not calls
        assert client.get("/").status_code == 200
        assert client.get("/").status_code == 200
    assert calls == ["registered"]
