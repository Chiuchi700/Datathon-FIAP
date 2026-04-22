import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_metadata():
    return {
        "seq_length": 60,
        "feature_cols": [
            "close",
            "return_1d",
            "ma_5_ratio",
            "ma_20_ratio",
            "volatility_10",
            "volume_zscore_20",
        ],
    }


@pytest.fixture
def sample_raw_df():
    """DataFrame com 100 linhas simulando dados brutos da Nike."""
    n = 100
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = np.cumprod(1 + rng.normal(0.0005, 0.015, n)) * 100.0
    volume = rng.integers(5_000_000, 15_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def sample_model_info():
    return {
        "model_name": "nike_lstm_forecaster",
        "model_alias": "candidate",
        "seq_length": 60,
        "feature_cols": [
            "close",
            "return_1d",
            "ma_5_ratio",
            "ma_20_ratio",
            "volatility_10",
            "volume_zscore_20",
        ],
    }


@pytest.fixture
def sample_predict_result():
    return {
        "predicted_return": 0.0042,
        "last_close": 95.5,
        "predicted_price": 95.9,
    }


@pytest.fixture
def sample_explain_result():
    return {
        "explanation": "O modelo preve alta moderada para a Nike amanha.",
        "close": 95.5,
        "predicted_price": 95.9,
        "predicted_return": 0.0042,
    }


@pytest.fixture
def sample_chat_result():
    return {
        "answer": "O modelo usa LSTM com 60 dias de janela temporal.",
        "question": "Qual o seq_length do modelo?",
    }
