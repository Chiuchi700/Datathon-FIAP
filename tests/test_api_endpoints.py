"""
Testes unitários para os endpoints da API FastAPI (src/api/app.py).

Todas as dependências externas (MLflow, dados, modelo) são mockadas
para garantir isolamento e execução sem infraestrutura.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200


def test_root_returns_expected_message():
    response = client.get("/")
    assert response.json() == {"message": "Nike Forecast API está no ar"}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_get_model_info(sample_model_info):
    with patch("src.api.app.get_model_info", return_value=sample_model_info) as mock:
        yield mock


def test_model_info_returns_200(mock_get_model_info):
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_returns_correct_schema(mock_get_model_info, sample_model_info):
    response = client.get("/model-info")
    data = response.json()
    assert data["model_name"] == sample_model_info["model_name"]
    assert data["model_alias"] == sample_model_info["model_alias"]
    assert data["seq_length"] == sample_model_info["seq_length"]
    assert data["feature_cols"] == sample_model_info["feature_cols"]


def test_model_info_calls_service_once(mock_get_model_info):
    client.get("/model-info")
    mock_get_model_info.assert_called_once()


# ---------------------------------------------------------------------------
# POST /predict — cenários de sucesso
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_predict_from_close(sample_model_info, sample_predict_result):
    full_response = {
        **sample_model_info,
        **sample_predict_result,
    }
    with patch("src.api.app.predict_from_close", return_value=full_response) as mock:
        yield mock


def test_predict_returns_200(mock_predict_from_close):
    response = client.post("/predict", json={"close": 95.5})
    assert response.status_code == 200


def test_predict_response_contains_required_fields(mock_predict_from_close):
    response = client.post("/predict", json={"close": 95.5})
    data = response.json()
    required_fields = {
        "model_name",
        "model_alias",
        "seq_length",
        "feature_cols",
        "last_close",
        "predicted_return",
        "predicted_price",
    }
    assert required_fields.issubset(data.keys())


def test_predict_passes_close_to_service(mock_predict_from_close):
    client.post("/predict", json={"close": 123.45})
    mock_predict_from_close.assert_called_once_with(123.45)


def test_predict_response_values(mock_predict_from_close, sample_predict_result):
    response = client.post("/predict", json={"close": 95.5})
    data = response.json()
    assert data["predicted_return"] == pytest.approx(sample_predict_result["predicted_return"])
    assert data["predicted_price"] == pytest.approx(sample_predict_result["predicted_price"])
    assert data["last_close"] == pytest.approx(sample_predict_result["last_close"])


# ---------------------------------------------------------------------------
# POST /predict — cenários de erro
# ---------------------------------------------------------------------------


def test_predict_returns_400_on_value_error():
    with patch(
        "src.api.app.predict_from_close",
        side_effect=ValueError("Histórico insuficiente"),
    ):
        response = client.post("/predict", json={"close": 0.0})
    assert response.status_code == 400
    assert "Histórico insuficiente" in response.json()["detail"]


def test_predict_returns_500_on_generic_exception():
    with patch(
        "src.api.app.predict_from_close",
        side_effect=RuntimeError("Falha no modelo"),
    ):
        response = client.post("/predict", json={"close": 95.5})
    assert response.status_code == 500
    assert "Falha no modelo" in response.json()["detail"]


def test_predict_returns_422_when_close_missing():
    """Pydantic deve rejeitar body sem o campo obrigatório 'close'."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_returns_422_when_close_is_string():
    """'close' deve ser float; string inválida deve gerar 422."""
    response = client.post("/predict", json={"close": "não-é-número"})
    assert response.status_code == 422


def test_predict_returns_422_when_body_is_empty():
    response = client.post("/predict", content=b"", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_explain_deps(sample_model_info, sample_predict_result):
    full_predict = {**sample_model_info, **sample_predict_result}
    with (
        patch("src.api.app.predict_from_close", return_value=full_predict) as mock_predict,
        patch("src.api.app.explain_prediction", return_value="O modelo preve alta moderada.") as mock_explain,
    ):
        yield mock_predict, mock_explain


def test_explain_returns_200(mock_explain_deps):
    response = client.post("/explain", json={"close": 95.5})
    assert response.status_code == 200


def test_explain_response_contains_explanation(mock_explain_deps):
    response = client.post("/explain", json={"close": 95.5})
    data = response.json()
    assert data["explanation"] == "O modelo preve alta moderada."
    assert data["close"] == pytest.approx(95.5)
    assert "predicted_price" in data
    assert "predicted_return" in data


def test_explain_returns_400_on_value_error():
    with patch("src.api.app.predict_from_close", side_effect=ValueError("Historico insuficiente")):
        response = client.post("/explain", json={"close": 0.0})
    assert response.status_code == 400


def test_explain_returns_500_on_generic_exception():
    with patch("src.api.app.predict_from_close", side_effect=RuntimeError("Falha")):
        response = client.post("/explain", json={"close": 95.5})
    assert response.status_code == 500


def test_explain_returns_422_when_close_missing():
    response = client.post("/explain", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chat_deps():
    with patch("src.api.app.chat_about_model", return_value="O modelo usa LSTM bidirecional.") as mock:
        yield mock


def test_chat_returns_200(mock_chat_deps):
    response = client.post("/chat", json={"question": "Qual modelo e usado?"})
    assert response.status_code == 200


def test_chat_response_contains_answer(mock_chat_deps):
    response = client.post("/chat", json={"question": "Qual modelo e usado?"})
    data = response.json()
    assert data["answer"] == "O modelo usa LSTM bidirecional."
    assert data["question"] == "Qual modelo e usado?"


def test_chat_returns_500_on_exception():
    with patch("src.api.app.chat_about_model", side_effect=RuntimeError("Erro")):
        response = client.post("/chat", json={"question": "Algo?"})
    assert response.status_code == 500


def test_chat_returns_422_when_question_missing():
    response = client.post("/chat", json={})
    assert response.status_code == 422
