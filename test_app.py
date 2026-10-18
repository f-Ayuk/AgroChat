
"""Tests for the ChatAgro Flask application."""

import pytest

from app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "ok"
    assert data["knowledge_base_size"] > 0
    assert data["embedder"]
    assert isinstance(data["similarity_threshold"], (int, float))


def test_chat_page_loads_script(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"/static/style.js" in response.data

    js_response = client.get("/static/style.js")

    assert js_response.status_code == 200
    assert b"addMessage(data.answer" in js_response.data


def test_empty_question_rejected(client):
    response = client.post(
        "/api/chat",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_missing_question_rejected(client):
    response = client.post("/api/chat", json={})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_long_question_rejected(client):
    response = client.post(
        "/api/chat",
        json={"question": "a" * 501},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_text_question_rejected(client):
    response = client.post(
        "/api/chat",
        json={"question": 123},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


@pytest.mark.parametrize(
    "question",
    [
        "How can I improve soil fertility?",
        "How can farmers control crop pests?",
    ],
)
def test_agriculture_question_returns_result(client, question):
    response = client.post(
        "/api/chat",
        json={"question": question},
    )

    assert response.status_code == 200

    data = response.get_json()

    assert isinstance(data["answer"], str)
    assert data["answer"].strip()
    assert isinstance(data["similarity"], (int, float))
    assert isinstance(data["accepted"], bool)
    assert isinstance(data["response_time_ms"], (int, float))


def test_form_post_works_without_js(client):
    question = "How can I improve soil fertility?"

    response = client.post(
        "/",
        data={"question": question},
    )

    assert response.status_code == 200
    assert question.encode() in response.data


def test_form_rejects_empty_question(client):
    response = client.post(
        "/",
        data={"question": "   "},
    )

    assert response.status_code == 400


def test_form_rejects_long_question(client):
    response = client.post(
        "/",
        data={"question": "a" * 501},
    )

    assert response.status_code == 400