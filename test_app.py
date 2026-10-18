from app import app


def test_health():
    client = app.test_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok"
    assert r.json["knowledge_base_size"] > 0


def test_chat_page_loads_script():
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"/static/style.js" in page.data
    js = client.get("/static/style.js")
    assert js.status_code == 200
    assert b"addMessage(data.answer" in js.data


def test_empty_question_rejected():
    client = app.test_client()
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400


def test_long_question_rejected():
    client = app.test_client()
    assert client.post(
        "/api/chat", json={"question": "a" * 501}
    ).status_code == 400


def test_non_text_rejected():
    client = app.test_client()
    assert client.post("/api/chat", json={"question": 123}).status_code == 400


def test_agriculture_question_returns_result():
    client = app.test_client()
    r = client.post(
        "/api/chat",
        json={"question": "How can I improve soil fertility?"},
    )
    assert r.status_code == 200
    assert r.json["answer"]
    assert "similarity" in r.json
    assert "response_time_ms" in r.json


def test_form_post_works_without_js():
    client = app.test_client()
    r = client.post(
        "/", data={"question": "How can I improve soil fertility?"}
    )
    assert r.status_code == 200
    assert b"How can I improve soil fertility?" in r.data