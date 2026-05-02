"""
Testes unitarios para o MCP Server (src/mcp_server.py).

Todas as chamadas HTTP a FastAPI sao mockadas.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.mcp_server import _call_api, call_tool


# ---------------------------------------------------------------------------
# _call_api
# ---------------------------------------------------------------------------


class TestCallApi:
    @patch("src.mcp_server.httpx.Client")
    def test_get_returns_json_string(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_response))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = _call_api("GET", "/health")
        parsed = json.loads(result)
        assert parsed["status"] == "ok"

    @patch("src.mcp_server.httpx.Client")
    def test_post_sends_payload(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"predicted_price": 96.0}
        mock_response.raise_for_status = MagicMock()
        mock_post = MagicMock(return_value=mock_response)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=mock_post)
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = _call_api("POST", "/predict", {"close": 95.5})
        parsed = json.loads(result)
        assert parsed["predicted_price"] == 96.0

    @patch("src.mcp_server.httpx.Client")
    def test_raises_on_http_error(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_response))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(httpx.HTTPStatusError):
            _call_api("GET", "/not-found")


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


class TestCallTool:
    @pytest.mark.asyncio
    @patch("src.mcp_server._call_api")
    async def test_predict_tool(self, mock_api):
        mock_api.return_value = '{"predicted_price": 96.0}'
        result = await call_tool("predict", {"close": 95.5})
        assert len(result) == 1
        assert "96.0" in result[0].text
        mock_api.assert_called_once_with("POST", "/predict", {"close": 95.5})

    @pytest.mark.asyncio
    @patch("src.mcp_server._call_api")
    async def test_model_info_tool(self, mock_api):
        mock_api.return_value = '{"model_name": "nike_lstm_forecaster"}'
        result = await call_tool("model_info", {})
        assert "nike_lstm_forecaster" in result[0].text
        mock_api.assert_called_once_with("GET", "/model-info")

    @pytest.mark.asyncio
    @patch("src.mcp_server._call_api")
    async def test_explain_tool(self, mock_api):
        mock_api.return_value = '{"explanation": "A Nike deve subir."}'
        result = await call_tool("explain_prediction", {"close": 95.5})
        assert "Nike" in result[0].text
        mock_api.assert_called_once_with("POST", "/explain", {"close": 95.5})

    @pytest.mark.asyncio
    @patch("src.mcp_server._call_api")
    async def test_ask_tool(self, mock_api):
        mock_api.return_value = '{"answer": "O modelo usa LSTM."}'
        result = await call_tool("ask_about_model", {"question": "Qual modelo?"})
        assert "LSTM" in result[0].text
        mock_api.assert_called_once_with("POST", "/chat", {"question": "Qual modelo?"})

    @pytest.mark.asyncio
    @patch("src.mcp_server._call_api")
    async def test_unknown_tool(self, mock_api):
        result = await call_tool("tool_inexistente", {})
        assert "error" in result[0].text
        mock_api.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.mcp_server._call_api")
    async def test_handles_api_error(self, mock_api):
        mock_api.side_effect = httpx.ConnectError("Connection refused")
        result = await call_tool("predict", {"close": 95.5})
        assert "error" in result[0].text.lower() or "Erro" in result[0].text
