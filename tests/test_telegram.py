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
        if request.url.path.endswith("/getWebhookInfo"):
            return httpx.Response(200, json={"ok": True, "result": {"url": ""}})
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


def test_existing_webhook_skips_repeated_registration(monkeypatch):
    import asyncio

    import httpx

    from rag_demo.telegram import register_cloud_webhook

    calls = []

    def reply(request):
        name = request.url.path.rsplit("/", 1)[-1]
        calls.append(name)
        if name == "getWebhookInfo":
            return httpx.Response(200, json={
                "ok": True,
                "result": {
                    "url": "https://rag-demo-ruddy.vercel.app/integrations/telegram/webhook"
                },
            })
        return httpx.Response(200, json={"ok": True, "result": True})

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(reply), **kwargs)

    monkeypatch.setattr("rag_demo.telegram.httpx.AsyncClient", client_factory)
    settings = Settings(
        _env_file=None, telegram_bot_token="123:fake", telegram_chat_id="789"
    )
    assert asyncio.run(register_cloud_webhook(settings))
    assert calls == ["getWebhookInfo"]
    force = settings.model_copy(update={"telegram_force_register": True})
    assert asyncio.run(register_cloud_webhook(force))
    assert calls[-1] == "setWebhook"


def test_telegram_http_url_logging_is_disabled():
    import logging

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING


def test_telegram_default_answer_is_grounded_not_a_source_dump():
    """Public no-LLM demo replies from actual FAQ, not arbitrary hash hits."""
    settings = Settings(
        _env_file=None, store_backend="memory", demo_mode=True,
        telegram_bot_token="123:example", telegram_chat_id="789",
    )
    with TestClient(create_app(settings)) as client:
        for query in ("What does Tayoca do?", "/ask What does Tayoca do?"):
            response = client.post(
                "/integrations/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:example")},
                json={"update_id": 3, "message": {"chat": {"id": 789}, "text": query}},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["method"] == "sendMessage"
            assert data["chat_id"] == "789"
            assert "engineering-led technology company" in data["text"]
            assert "Cloud cost and FinOps" in data["text"]
            assert "Source: https://tayoca.com/" in data["text"]
            assert "Operator Brief" not in data["text"]
            assert len(data["text"]) < 750


def test_telegram_public_faq_and_unknown_question_fail_closed():
    settings = Settings(
        _env_file=None, store_backend="memory", demo_mode=True,
        telegram_bot_token="123:example", telegram_chat_id="789",
    )
    with TestClient(create_app(settings)) as client:
        for query, expected in (
            ("What is Sivanta?", "conversation-led commerce automation"),
            ("What services does Tayoca provide?", "reliability and DevSecOps"),
            ("What is the Operator Brief?", "recurring briefing"),
            ("How can I contact Tayoca?", "support@tayoca.com"),
        ):
            response = client.post(
                "/integrations/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:example")},
                json={"update_id": 9, "message": {"chat": {"id": 789}, "text": query}},
            )
            assert response.status_code == 200
            assert expected in response.json()["text"]

        unknown = client.post(
            "/integrations/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:example")},
            json={"update_id": 10, "message": {
                "chat": {"id": 789}, "text": "How can I tune a solar inverter?"
            }},
        )
        assert unknown.status_code == 200
        assert "couldn't find a reliable answer" in unknown.json()["text"]
        assert "Operator Brief" not in unknown.json()["text"]


def test_telegram_explicit_search_preserves_source_excerpts():
    settings = Settings(
        _env_file=None, store_backend="memory", demo_mode=True,
        telegram_bot_token="123:example", telegram_chat_id="789",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/integrations/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": webhook_secret("123:example")},
            json={"update_id": 4, "message": {
                "chat": {"id": 789}, "text": "/search What does Tayoca do?"
            }},
        )
        assert response.status_code == 200
        assert response.json()["text"].startswith(
            "Retrieved source excerpts (not generated answers):"
        )
