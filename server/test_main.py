import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import main

OLLAMA_CHAT = f"{main.OLLAMA_URL}/api/chat"
OLLAMA_VERSION = f"{main.OLLAMA_URL}/api/version"
BODY = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def ollama():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


def ndjson(*chunks):
    return "\n".join(json.dumps(c) for c in chunks) + "\n"


def test_health_reports_ollama_up(client, ollama):
    ollama.get(OLLAMA_VERSION).respond(200, json={"version": "1"})
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "ollama": True}


def test_health_reports_ollama_down(client, ollama):
    ollama.get(OLLAMA_VERSION).mock(side_effect=httpx.ConnectError("refused"))
    assert client.get("/health").json() == {"status": "ok", "ollama": False}


def test_swagger_ui_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/chat", "/chat-stream"} <= paths.keys()


def test_chat_forwards_payload_and_returns_message(client, ollama):
    route = ollama.post(OLLAMA_CHAT).respond(
        200,
        json={"model": "m1", "message": {"role": "assistant", "content": "hello"}},
    )
    res = client.post("/chat", json={**BODY, "model": "m1"})

    assert res.status_code == 200
    assert res.json() == {"model": "m1", "message": {"role": "assistant", "content": "hello"}}
    sent = json.loads(route.calls.last.request.content)
    assert sent == {"model": "m1", "messages": BODY["messages"], "stream": False}


def test_chat_uses_default_model(client, ollama):
    route = ollama.post(OLLAMA_CHAT).respond(
        200, json={"model": main.DEFAULT_MODEL, "message": {"role": "assistant", "content": "x"}}
    )
    client.post("/chat", json=BODY)
    assert json.loads(route.calls.last.request.content)["model"] == main.DEFAULT_MODEL


def test_chat_propagates_ollama_error_status(client, ollama):
    ollama.post(OLLAMA_CHAT).respond(404, text="model not found")
    res = client.post("/chat", json=BODY)
    assert res.status_code == 404
    assert "model not found" in res.json()["detail"]


def test_chat_returns_502_when_ollama_unreachable(client, ollama):
    ollama.post(OLLAMA_CHAT).mock(side_effect=httpx.ConnectError("refused"))
    assert client.post("/chat", json=BODY).status_code == 502


@pytest.mark.parametrize(
    "body",
    [{}, {"messages": [{"role": "robot", "content": "x"}]}, {"messages": [{"role": "user"}]}],
)
def test_endpoints_validate_payload(client, body):
    assert client.post("/chat", json=body).status_code == 422
    assert client.post("/chat-stream", json=body).status_code == 422


def test_chat_stream_yields_tokens(client, ollama):
    route = ollama.post(OLLAMA_CHAT).respond(
        200,
        text=ndjson(
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ),
    )
    res = client.post("/chat-stream", json=BODY)

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert res.text == "Hello"
    assert json.loads(route.calls.last.request.content)["stream"] is True


def test_chat_stream_reports_ollama_error_chunk(client, ollama):
    ollama.post(OLLAMA_CHAT).respond(
        200, text=ndjson({"message": {"content": "a"}}, {"error": "boom"})
    )
    assert client.post("/chat-stream", json=BODY).text == "a[error] boom"


def test_chat_stream_reports_http_error(client, ollama):
    ollama.post(OLLAMA_CHAT).respond(500, text="server exploded")
    assert client.post("/chat-stream", json=BODY).text == "[error] server exploded"


def test_chat_stream_reports_unreachable(client, ollama):
    ollama.post(OLLAMA_CHAT).mock(side_effect=httpx.ConnectError("refused"))
    assert client.post("/chat-stream", json=BODY).text.startswith("[error] Ollama unreachable")
