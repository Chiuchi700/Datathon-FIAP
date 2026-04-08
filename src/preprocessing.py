from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS_DEFAULT: list[str] = [
    "close",
    "return_1d",
    "ma_5_ratio",
    "ma_20_ratio",
    "volatility_10",
    "volume_zscore_20",
]



def create_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        "_".join(str(c) for c in col if c is not None).lower()
        if isinstance(col, tuple)
        else str(col).lower()
        for col in df.columns
    ]

    rename_map: dict[str, str] = {}
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

    required_cols = ["close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {missing_cols}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)

    df["return_1d"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_5_ratio"] = df["close"] / df["ma_5"] - 1.0
    df["ma_20_ratio"] = df["close"] / df["ma_20"] - 1.0
    df["volatility_10"] = df["return_1d"].rolling(10).std()

    if "volume" in df.columns:
        vol_mean_20 = df["volume"].rolling(20).mean()
        vol_std_20 = df["volume"].rolling(20).std()
        df["volume_zscore_20"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)
    else:
        df["volume_zscore_20"] = 0.0

    df["target_return_1d"] = df["close"].shift(-1) / df["close"] - 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df



def save_processed_data(df: pd.DataFrame, processed_path: str | Path) -> Path:
    processed_path = Path(processed_path)
    os.makedirs(processed_path.parent, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return processed_path



def load_processed_data(processed_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(processed_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df



def _build_sequences(
    features_scaled: NDArray[np.float32] | NDArray[np.float64],
    close_values: NDArray[np.float32] | NDArray[np.float64],
    target_returns: NDArray[np.float32] | NDArray[np.float64],
    dates: NDArray[Any],
    seq_length: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[Any],
]:
    X: list[NDArray[np.float32] | NDArray[np.float64]] = []
    y: list[float] = []
    prev_close: list[float] = []
    target_date: list[Any] = []

    for i in range(seq_length, len(features_scaled)):
        X.append(features_scaled[i - seq_length:i])
        y.append(float(target_returns[i]))
        prev_close.append(float(close_values[i]))
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
    feature_cols: list[str] | None = None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    dict[str, Any],
]:
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT

    required_cols = feature_cols + ["close", "target_return_1d"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {missing_cols}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    data = df.copy().reset_index(drop=True)
    n = len(data)
    if n <= seq_length + 10:
        raise ValueError("Poucos dados para criar sequências. Aumente a base ou reduza seq_length.")

    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    scaler = MinMaxScaler()
    scaler.fit(data.loc[: train_end - 1, feature_cols])

    features_scaled = scaler.transform(data[feature_cols])
    close_values = data["close"].values
    target_returns = data["target_return_1d"].values
    dates = data["date"].values if "date" in data.columns else np.arange(n)

    X_all, y_all, prev_close_all, target_dates_all = _build_sequences(
        features_scaled,
        close_values,
        target_returns,
        dates,
        seq_length,
    )

    sequence_end_indices = np.arange(seq_length, n)

    train_mask = sequence_end_indices < train_end
    val_mask = (sequence_end_indices >= train_end) & (sequence_end_indices < val_end)
    test_mask = sequence_end_indices >= val_end

    artifacts: dict[str, Any] = {
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
