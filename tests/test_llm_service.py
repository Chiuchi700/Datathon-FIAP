"""
Testes unitarios para o servico LLM (src/api/llm_service.py).

Todas as chamadas ao Ollama sao mockadas.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.api.llm_service import (
    _FALLBACK_MESSAGE,
    explain_prediction,
    generate_text,
)


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------


class TestGenerateText:
    @patch("src.api.llm_service.httpx.Client")
    def test_returns_response_text_on_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Nike vai subir amanha"}
        mock_response.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_response)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = generate_text("Explique a previsao")
        assert result == "Nike vai subir amanha"

    @patch("src.api.llm_service.httpx.Client")
    def test_returns_fallback_when_ollama_unavailable(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(side_effect=httpx.ConnectError("Connection refused")))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = generate_text("qualquer prompt")
        assert result == _FALLBACK_MESSAGE

    @patch("src.api.llm_service.httpx.Client")
    def test_returns_fallback_on_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(side_effect=httpx.TimeoutException("timeout")))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = generate_text("qualquer prompt")
        assert result == _FALLBACK_MESSAGE

    @patch("src.api.llm_service.httpx.Client")
    def test_sends_correct_payload(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_post = MagicMock(return_value=mock_response)
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=mock_post))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        generate_text("meu prompt")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["prompt"] == "meu prompt"
        assert payload["stream"] is False

    @patch("src.api.llm_service.httpx.Client")
    def test_returns_fallback_on_missing_response_key(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "model not found"}
        mock_response.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_response)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = generate_text("prompt")
        assert result == _FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# explain_prediction
# ---------------------------------------------------------------------------


class TestExplainPrediction:
    @patch("src.api.llm_service.generate_text")
    def test_returns_explanation_string(self, mock_gen):
        mock_gen.return_value = "A Nike deve subir levemente."
        result = explain_prediction(close=95.5, predicted_price=95.9, predicted_return=0.0042)
        assert result == "A Nike deve subir levemente."

    @patch("src.api.llm_service.generate_text")
    def test_prompt_contains_prediction_data(self, mock_gen):
        mock_gen.return_value = "ok"
        explain_prediction(close=95.5, predicted_price=95.9, predicted_return=0.0042)

        prompt = mock_gen.call_args[0][0]
        assert "95.50" in prompt
        assert "95.90" in prompt
        assert "0.42" in prompt  # 0.0042 formatado como percentual

    @patch("src.api.llm_service.generate_text")
    def test_prompt_shows_alta_for_positive_return(self, mock_gen):
        mock_gen.return_value = "ok"
        explain_prediction(close=95.0, predicted_price=96.0, predicted_return=0.01)
        prompt = mock_gen.call_args[0][0]
        assert "alta" in prompt

    @patch("src.api.llm_service.generate_text")
    def test_prompt_shows_queda_for_negative_return(self, mock_gen):
        mock_gen.return_value = "ok"
        explain_prediction(close=95.0, predicted_price=94.0, predicted_return=-0.01)
        prompt = mock_gen.call_args[0][0]
        assert "queda" in prompt

    @patch("src.api.llm_service.generate_text")
    def test_handles_llm_fallback(self, mock_gen):
        mock_gen.return_value = _FALLBACK_MESSAGE
        result = explain_prediction(close=95.5, predicted_price=95.9, predicted_return=0.0042)
        assert result == _FALLBACK_MESSAGE
