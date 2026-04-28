import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Caminho base do projeto no Docker
PROJECT_ROOT = Path("/opt/airflow/project")

FEATURE_COLS_DEFAULT = [
    "close",
    "return_1d",
    "ma_5_ratio",
    "ma_20_ratio",
    "volatility_10",
    "volume_zscore_20",
]


def create_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza a limpeza e feature engineering dos dados da Nike."""
    df = df.copy()

    # Normaliza nomes de colunas (trata MultiIndex do yfinance se houver)
    df.columns = [
        (
            "_".join(str(c) for c in col if c is not None).lower()
            if isinstance(col, tuple)
            else str(col).lower()
        )
        for col in df.columns
    ]

    # Mapeamento para garantir nomes padrão
    rename_map = {}
    for col in df.columns:
        if "date" in col:
            rename_map[col] = "date"
        elif "open" in col:
            rename_map[col] = "open"
        elif "high" in col:
            rename_map[col] = "high"
        elif "low" in col:
            rename_map[col] = "low"
        elif "close" in col:
            rename_map[col] = "close"
        elif "volume" in col:
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    if "close" not in df.columns:
        raise ValueError(
            f"Coluna 'close' não encontrada. Colunas: {df.columns.tolist()}"
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    # Conversão numérica e limpeza
    numeric_cols = [
        c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Feature Engineering
    df["return_1d"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_5_ratio"] = df["close"] / df["ma_5"] - 1.0
    df["ma_20_ratio"] = df["close"] / df["ma_20"] - 1.0
    df["volatility_10"] = df["return_1d"].rolling(10).std()

    if "volume" in df.columns:
        vol_mean_20 = df["volume"].rolling(20).mean()
        vol_std_20 = df["volume"].rolling(20).std()
        df["volume_zscore_20"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(
            0, np.nan
        )
    else:
        df["volume_zscore_20"] = 0.0

    # Target: Retorno do dia seguinte
    df["target_return_1d"] = df["close"].shift(-1) / df["close"] - 1.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def save_processed_data(df: pd.DataFrame, processed_path: str | Path) -> None:
    """Salva os dados processados garantindo a criação da pasta."""
    path = Path(processed_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed_data(processed_path: str | Path) -> pd.DataFrame:
    """Carrega os dados processados e converte datas."""
    path = Path(processed_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _build_sequences(features_scaled, close_values, target_returns, dates, seq_length):
    """Cria as janelas deslizantes para o modelo LSTM."""
    X, y, prev_close, target_date = [], [], [], []
    for i in range(seq_length, len(features_scaled)):
        X.append(features_scaled[i - seq_length : i])
        y.append(target_returns[i])
        prev_close.append(close_values[i])
        target_date.append(dates[i])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(prev_close, dtype=np.float32),
        np.array(target_date),
    )


def prepare_sequences(
    df: pd.DataFrame,
    seq_length: int = 60,
    train_split: float = 0.70,
    val_split: float = 0.15,
    feature_cols=None,
):
    """Normaliza os dados e divide em conjuntos de treino, validação e teste."""
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT
    data = df.copy().reset_index(drop=True)
    n = len(data)

    if n <= seq_length + 10:
        raise ValueError("Dados insuficientes para a janela (seq_length).")

    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    # O Fit do Scaler deve ocorrer apenas nos dados de treino para evitar data leakage
    scaler = MinMaxScaler()
    scaler.fit(data.loc[: train_end - 1, feature_cols])

    features_scaled = scaler.transform(data[feature_cols])
    close_values = data["close"].values
    target_returns = data["target_return_1d"].values
    dates = data["date"].values if "date" in data.columns else np.arange(n)

    X_all, y_all, prev_close_all, target_dates_all = _build_sequences(
        features_scaled, close_values, target_returns, dates, seq_length
    )

    # Máscaras para divisão temporal
    sequence_end_indices = np.arange(seq_length, n)
    train_mask = sequence_end_indices < train_end
    val_mask = (sequence_end_indices >= train_end) & (sequence_end_indices < val_end)
    test_mask = sequence_end_indices >= val_end

    artifacts = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_length": seq_length,
    }

    return (
        X_all[train_mask],
        X_all[val_mask],
        X_all[test_mask],
        y_all[train_mask],
        y_all[val_mask],
        y_all[test_mask],
        prev_close_all[train_mask],
        prev_close_all[val_mask],
        prev_close_all[test_mask],
        target_dates_all[train_mask],
        target_dates_all[val_mask],
        target_dates_all[test_mask],
        artifacts,
    )
