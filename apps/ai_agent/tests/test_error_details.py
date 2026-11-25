"""Internal error details are not leaked by /index/message and /search (issue #711)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

_SECRET = "weaviate-internal.cluster.local:8080 refused connection"

_INDEX_BODY = {
    "messageId": "msg-1",
    "conversationId": "conv-1",
    "senderId": "user-1",
    "content": "hello",
}


def _call(endpoint: str):
    if endpoint == "index":
        return client.post("/index/message", json=_INDEX_BODY)
    return client.get("/search", params={"q": "hello", "conversationId": "conv-1"})


@pytest.mark.parametrize("endpoint", ["index", "search"])
def test_backend_error_returns_fixed_503_message(endpoint):
    mock_wv = MagicMock()
    mock_wv.collections.exists.side_effect = RuntimeError(_SECRET)

    with patch("main.weaviate.connect_to_local", return_value=mock_wv):
        response = _call(endpoint)

    assert response.status_code == 503
    assert response.json() == {"detail": "Search backend unavailable"}
    assert _SECRET not in response.text
    mock_wv.close.assert_called_once()


@pytest.mark.parametrize("endpoint", ["index", "search"])
def test_backend_error_is_logged(endpoint, caplog):
    mock_wv = MagicMock()
    mock_wv.collections.exists.side_effect = RuntimeError(_SECRET)

    with patch("main.weaviate.connect_to_local", return_value=mock_wv):
        _call(endpoint)

    assert any(_SECRET in (r.exc_text or "") or r.exc_info for r in caplog.records)


@pytest.mark.parametrize("endpoint", ["index", "search"])
def test_missing_api_key_still_returns_500(endpoint, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_wv = MagicMock()
    mock_wv.collections.exists.return_value = True

    with patch("main.weaviate.connect_to_local", return_value=mock_wv):
        response = _call(endpoint)

    assert response.status_code == 500
    assert response.json() == {"detail": "OPENAI_API_KEY is not configured"}
