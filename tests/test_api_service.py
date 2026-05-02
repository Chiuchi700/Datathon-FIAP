"""
Testes unitários para a camada de serviço da API (src/api/service.py).

As funções são testadas com dependências externas mockadas:
  - download_metadata_from_registry  (MLflow)
  - load_raw_data                    (I/O de arquivo)
  - predict_next_day                 (inferência do modelo)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.api.service import (
    _build_feature_window_from_close,
    _normalize_raw_data,
    get_model_info,
    predict_from_close,
)

# ---------------------------------------------------------------------------
# _normalize_raw_data
# ---------------------------------------------------------------------------


class TestNormalizeRawData:
    def test_lowercase_columns(self):
        df = pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})
        result = _normalize_raw_data(df)
        assert "date" in result.columns
        assert "close" in result.columns

    def test_renames_columns_with_keywords(self):
        # Nomes devem conter as substrings exatas que o service verifica
        # ("date", "close", "volume") após lowercase
        df = pd.DataFrame(
            {
                "Price Date": ["2024-01-01"],
                "Adj Close": [100.0],
                "NKE Volume": [1_000_000.0],
            }
        )
        result = _normalize_raw_data(df)
        assert "date" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_raises_when_date_column_missing(self):
        df = pd.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
            _normalize_raw_data(df)

    def test_raises_when_close_column_missing(self):
        df = pd.DataFrame({"date": ["2024-01-01"]})
        with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
            _normalize_raw_data(df)

    def test_raises_when_all_close_values_are_nan(self):
        df = pd.DataFrame({"date": ["2024-01-01"], "close": [None]})
        with pytest.raises(ValueError, match="Não foi possível montar histórico"):
            _normalize_raw_data(df)

    def test_converts_close_to_numeric(self, sample_raw_df):
        df = sample_raw_df.copy()
        df["close"] = df["close"].astype(str)
        result = _normalize_raw_data(df)
        assert pd.api.types.is_float_dtype(result["close"])

    def test_removes_duplicate_columns(self):
        df = pd.DataFrame({"date": ["2024-01-01"], "close": [100.0], "close_adj": [99.0]})
        # Ambas têm "close" no nome → apenas a primeira deve ficar após deduplicação
        result = _normalize_raw_data(df)
        assert result.columns.tolist().count("close") == 1

    def test_sorts_by_date(self, sample_raw_df):
        shuffled = sample_raw_df.sample(frac=1, random_state=0).reset_index(drop=True)
        result = _normalize_raw_data(shuffled)
        assert result["date"].is_monotonic_increasing

    def test_drops_rows_with_nan_close(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "close": [100.0, None, 102.0],
            }
        )
        result = _normalize_raw_data(df)
        assert len(result) == 2
        assert result["close"].notna().all()


# ---------------------------------------------------------------------------
# _build_feature_window_from_close
# ---------------------------------------------------------------------------


class TestBuildFeatureWindowFromClose:
    FEATURE_COLS = [
        "close",
        "return_1d",
        "ma_5_ratio",
        "ma_20_ratio",
        "volatility_10",
        "volume_zscore_20",
    ]
    SEQ_LENGTH = 60

    @patch("src.api.service.load_raw_data")
    def test_returns_window_with_correct_shape(self, mock_load, sample_raw_df):
        mock_load.return_value = sample_raw_df
        window, _ = _build_feature_window_from_close(
            close=95.0,
            feature_cols=self.FEATURE_COLS,
            seq_length=self.SEQ_LENGTH,
        )
        assert window.shape == (self.SEQ_LENGTH, len(self.FEATURE_COLS))

    @patch("src.api.service.load_raw_data")
    def test_returns_last_close_as_float(self, mock_load, sample_raw_df):
        mock_load.return_value = sample_raw_df
        _, last_close = _build_feature_window_from_close(
            close=95.0,
            feature_cols=self.FEATURE_COLS,
            seq_length=self.SEQ_LENGTH,
        )
        assert isinstance(last_close, float)

    @patch("src.api.service.load_raw_data")
    def test_last_close_equals_input_close(self, mock_load, sample_raw_df):
        """A última linha concatenada tem close = valor passado."""
        mock_load.return_value = sample_raw_df
        close_input = 123.45
        _, last_close = _build_feature_window_from_close(
            close=close_input,
            feature_cols=self.FEATURE_COLS,
            seq_length=self.SEQ_LENGTH,
        )
        assert last_close == pytest.approx(close_input)

    @patch("src.api.service.load_raw_data")
    def test_raises_when_history_insufficient(self, mock_load):
        """Menos de 25 linhas no histórico deve levantar ValueError."""
        small_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="B"),
                "close": np.linspace(90, 100, 10),
                "volume": [1_000_000.0] * 10,
            }
        )
        mock_load.return_value = small_df
        with pytest.raises(ValueError, match="Histórico insuficiente"):
            _build_feature_window_from_close(
                close=95.0,
                feature_cols=self.FEATURE_COLS,
                seq_length=self.SEQ_LENGTH,
            )

    @patch("src.api.service.load_raw_data")
    def test_raises_when_not_enough_rows_for_seq_length(self, mock_load):
        """Histórico com > 25 linhas mas < seq_length após feature generation."""
        n = 30  # > 25 mas < 60
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="B"),
                "close": np.linspace(90, 100, n),
                "volume": [1_000_000.0] * n,
            }
        )
        mock_load.return_value = df
        with pytest.raises(ValueError, match="não há linhas suficientes"):
            _build_feature_window_from_close(
                close=95.0,
                feature_cols=self.FEATURE_COLS,
                seq_length=self.SEQ_LENGTH,
            )

    @patch("src.api.service.load_raw_data")
    def test_window_contains_no_nan(self, mock_load, sample_raw_df):
        mock_load.return_value = sample_raw_df
        window, _ = _build_feature_window_from_close(
            close=95.0,
            feature_cols=self.FEATURE_COLS,
            seq_length=self.SEQ_LENGTH,
        )
        assert not np.isnan(window).any()


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    @patch("src.api.service.download_metadata_from_registry")
    def test_returns_correct_keys(self, mock_download, sample_metadata):
        mock_download.return_value = sample_metadata
        result = get_model_info()
        assert set(result.keys()) == {"model_name", "model_alias", "seq_length", "feature_cols"}

    @patch("src.api.service.download_metadata_from_registry")
    def test_uses_env_defaults(self, mock_download, sample_metadata):
        mock_download.return_value = sample_metadata
        result = get_model_info()
        assert result["model_name"] == "nike_lstm_forecaster"
        assert result["model_alias"] == "candidate"

    @patch("src.api.service.download_metadata_from_registry")
    def test_forwards_seq_length_from_metadata(self, mock_download, sample_metadata):
        mock_download.return_value = sample_metadata
        result = get_model_info()
        assert result["seq_length"] == sample_metadata["seq_length"]

    @patch("src.api.service.download_metadata_from_registry")
    def test_forwards_feature_cols_from_metadata(self, mock_download, sample_metadata):
        mock_download.return_value = sample_metadata
        result = get_model_info()
        assert result["feature_cols"] == sample_metadata["feature_cols"]

    @patch("src.api.service.download_metadata_from_registry")
    def test_calls_registry_with_correct_args(self, mock_download, sample_metadata):
        mock_download.return_value = sample_metadata
        get_model_info()
        mock_download.assert_called_once_with(
            model_name="nike_lstm_forecaster",
            alias="candidate",
        )


# ---------------------------------------------------------------------------
# predict_from_close
# ---------------------------------------------------------------------------


class TestPredictFromClose:
    FEATURE_COLS = [
        "close",
        "return_1d",
        "ma_5_ratio",
        "ma_20_ratio",
        "volatility_10",
        "volume_zscore_20",
    ]
    SEQ_LENGTH = 60

    def _dummy_window(self):
        return np.zeros((self.SEQ_LENGTH, len(self.FEATURE_COLS)))

    @patch("src.api.service.predict_next_day")
    @patch("src.api.service._build_feature_window_from_close")
    @patch("src.api.service.download_metadata_from_registry")
    def test_returns_correct_keys(
        self, mock_meta, mock_window, mock_predict, sample_metadata, sample_predict_result
    ):
        mock_meta.return_value = sample_metadata
        mock_window.return_value = (self._dummy_window(), 95.5)
        mock_predict.return_value = sample_predict_result

        result = predict_from_close(95.5)

        assert set(result.keys()) == {
            "model_name",
            "model_alias",
            "seq_length",
            "feature_cols",
            "last_close",
            "predicted_return",
            "predicted_price",
        }

    @patch("src.api.service.predict_next_day")
    @patch("src.api.service._build_feature_window_from_close")
    @patch("src.api.service.download_metadata_from_registry")
    def test_returns_prediction_values(
        self, mock_meta, mock_window, mock_predict, sample_metadata, sample_predict_result
    ):
        mock_meta.return_value = sample_metadata
        mock_window.return_value = (self._dummy_window(), 95.5)
        mock_predict.return_value = sample_predict_result

        result = predict_from_close(95.5)

        assert result["predicted_return"] == pytest.approx(sample_predict_result["predicted_return"])
        assert result["predicted_price"] == pytest.approx(sample_predict_result["predicted_price"])
        assert result["last_close"] == pytest.approx(sample_predict_result["last_close"])

    @patch("src.api.service.predict_next_day")
    @patch("src.api.service._build_feature_window_from_close")
    @patch("src.api.service.download_metadata_from_registry")
    def test_passes_close_to_feature_builder(
        self, mock_meta, mock_window, mock_predict, sample_metadata, sample_predict_result
    ):
        mock_meta.return_value = sample_metadata
        mock_window.return_value = (self._dummy_window(), 95.5)
        mock_predict.return_value = sample_predict_result

        predict_from_close(123.45)

        _, kwargs = mock_window.call_args
        assert kwargs.get("close") == 123.45 or mock_window.call_args[0][0] == 123.45

    @patch("src.api.service.predict_next_day")
    @patch("src.api.service._build_feature_window_from_close")
    @patch("src.api.service.download_metadata_from_registry")
    def test_propagates_value_error_from_feature_builder(
        self, mock_meta, mock_window, mock_predict, sample_metadata
    ):
        mock_meta.return_value = sample_metadata
        mock_window.side_effect = ValueError("Histórico insuficiente")

        with pytest.raises(ValueError, match="Histórico insuficiente"):
            predict_from_close(95.5)

    @patch("src.api.service.predict_next_day")
    @patch("src.api.service._build_feature_window_from_close")
    @patch("src.api.service.download_metadata_from_registry")
    def test_calls_predict_next_day_with_last_close(
        self, mock_meta, mock_window, mock_predict, sample_metadata, sample_predict_result
    ):
        mock_meta.return_value = sample_metadata
        mock_window.return_value = (self._dummy_window(), 95.5)
        mock_predict.return_value = sample_predict_result

        predict_from_close(95.5)

        mock_predict.assert_called_once()
        _, kwargs = mock_predict.call_args
        assert kwargs.get("last_close") == 95.5 or mock_predict.call_args[0][1] == 95.5
