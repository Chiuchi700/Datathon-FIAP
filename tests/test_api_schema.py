"""
Testes unitários para os schemas Pydantic da API (src/api/schema.py).
"""

import pytest
from pydantic import ValidationError

from src.api.schema import (
    ChatRequest,
    ChatResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureRow,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


# ---------------------------------------------------------------------------
# PredictRequest
# ---------------------------------------------------------------------------


class TestPredictRequest:
    def test_valid_close(self):
        req = PredictRequest(close=95.5)
        assert req.close == pytest.approx(95.5)

    def test_integer_coerced_to_float(self):
        req = PredictRequest(close=100)
        assert isinstance(req.close, float)

    def test_zero_is_valid(self):
        req = PredictRequest(close=0.0)
        assert req.close == 0.0

    def test_negative_close_is_valid(self):
        """Pydantic não impõe restrição de domínio — valor negativo deve ser aceito."""
        req = PredictRequest(close=-10.0)
        assert req.close == -10.0

    def test_missing_close_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest()

    def test_string_close_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest(close="não-é-número")

    def test_none_close_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest(close=None)


# ---------------------------------------------------------------------------
# PredictResponse
# ---------------------------------------------------------------------------


class TestPredictResponse:
    def _valid_payload(self):
        return {
            "model_name": "nike_lstm_forecaster",
            "model_alias": "candidate",
            "seq_length": 60,
            "feature_cols": ["close", "return_1d"],
            "last_close": 95.5,
            "predicted_return": 0.0042,
            "predicted_price": 95.9,
        }

    def test_valid_payload(self):
        resp = PredictResponse(**self._valid_payload())
        assert resp.model_name == "nike_lstm_forecaster"

    def test_missing_field_raises(self):
        payload = self._valid_payload()
        del payload["predicted_price"]
        with pytest.raises(ValidationError):
            PredictResponse(**payload)

    def test_feature_cols_is_list(self):
        resp = PredictResponse(**self._valid_payload())
        assert isinstance(resp.feature_cols, list)

    def test_seq_length_is_int(self):
        resp = PredictResponse(**self._valid_payload())
        assert isinstance(resp.seq_length, int)

    def test_predicted_return_precision(self):
        resp = PredictResponse(**self._valid_payload())
        assert resp.predicted_return == pytest.approx(0.0042)


# ---------------------------------------------------------------------------
# ModelInfoResponse
# ---------------------------------------------------------------------------


class TestModelInfoResponse:
    def _valid_payload(self):
        return {
            "model_name": "nike_lstm_forecaster",
            "model_alias": "candidate",
            "seq_length": 60,
            "feature_cols": ["close", "return_1d", "ma_5_ratio"],
        }

    def test_valid_payload(self):
        info = ModelInfoResponse(**self._valid_payload())
        assert info.seq_length == 60

    def test_missing_model_name_raises(self):
        payload = self._valid_payload()
        del payload["model_name"]
        with pytest.raises(ValidationError):
            ModelInfoResponse(**payload)

    def test_missing_feature_cols_raises(self):
        payload = self._valid_payload()
        del payload["feature_cols"]
        with pytest.raises(ValidationError):
            ModelInfoResponse(**payload)

    def test_empty_feature_cols_is_valid(self):
        payload = self._valid_payload()
        payload["feature_cols"] = []
        info = ModelInfoResponse(**payload)
        assert info.feature_cols == []


# ---------------------------------------------------------------------------
# FeatureRow
# ---------------------------------------------------------------------------


class TestFeatureRow:
    def _valid_payload(self):
        return {
            "close": 95.5,
            "return_1d": 0.005,
            "ma_5_ratio": 0.01,
            "ma_20_ratio": -0.02,
            "volatility_10": 0.03,
            "volume_zscore_20": 1.2,
        }

    def test_valid_payload(self):
        row = FeatureRow(**self._valid_payload())
        assert row.close == pytest.approx(95.5)

    def test_all_fields_required(self):
        for field in self._valid_payload():
            payload = self._valid_payload()
            del payload[field]
            with pytest.raises(ValidationError):
                FeatureRow(**payload)

    def test_negative_return_is_valid(self):
        payload = self._valid_payload()
        payload["return_1d"] = -0.05
        row = FeatureRow(**payload)
        assert row.return_1d == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# ExplainRequest
# ---------------------------------------------------------------------------


class TestExplainRequest:
    def test_valid_close(self):
        req = ExplainRequest(close=95.5)
        assert req.close == pytest.approx(95.5)

    def test_missing_close_raises(self):
        with pytest.raises(ValidationError):
            ExplainRequest()

    def test_string_close_raises(self):
        with pytest.raises(ValidationError):
            ExplainRequest(close="nao-e-numero")

    def test_integer_coerced_to_float(self):
        req = ExplainRequest(close=100)
        assert isinstance(req.close, float)


# ---------------------------------------------------------------------------
# ExplainResponse
# ---------------------------------------------------------------------------


class TestExplainResponse:
    def _valid_payload(self):
        return {
            "explanation": "O modelo preve alta.",
            "close": 95.5,
            "predicted_price": 95.9,
            "predicted_return": 0.0042,
        }

    def test_valid_payload(self):
        resp = ExplainResponse(**self._valid_payload())
        assert resp.explanation == "O modelo preve alta."

    def test_missing_explanation_raises(self):
        payload = self._valid_payload()
        del payload["explanation"]
        with pytest.raises(ValidationError):
            ExplainResponse(**payload)

    def test_missing_close_raises(self):
        payload = self._valid_payload()
        del payload["close"]
        with pytest.raises(ValidationError):
            ExplainResponse(**payload)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------


class TestChatRequest:
    def test_valid_question(self):
        req = ChatRequest(question="Qual o modelo?")
        assert req.question == "Qual o modelo?"

    def test_missing_question_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest()

    def test_empty_string_is_valid(self):
        req = ChatRequest(question="")
        assert req.question == ""


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------


class TestChatResponse:
    def _valid_payload(self):
        return {
            "answer": "O modelo usa LSTM.",
            "question": "Qual o modelo?",
        }

    def test_valid_payload(self):
        resp = ChatResponse(**self._valid_payload())
        assert resp.answer == "O modelo usa LSTM."

    def test_missing_answer_raises(self):
        payload = self._valid_payload()
        del payload["answer"]
        with pytest.raises(ValidationError):
            ChatResponse(**payload)

    def test_missing_question_raises(self):
        payload = self._valid_payload()
        del payload["question"]
        with pytest.raises(ValidationError):
            ChatResponse(**payload)
