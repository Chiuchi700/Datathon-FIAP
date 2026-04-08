import os

import pandas as pd

from src.data_loader import load_raw_data
from src.inference.inference import predict_next_day
from src.model_registry import download_metadata_from_registry

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "nike_lstm_forecaster")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "candidate")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/nike_raw.csv")


def get_model_info() -> dict:
    metadata = download_metadata_from_registry(
        model_name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    return {
        "model_name": REGISTERED_MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "seq_length": metadata["seq_length"],
        "feature_cols": metadata["feature_cols"],
    }


def _normalize_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df.columns = [str(col).strip().lower() for col in df.columns]

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

    required_cols = ["date", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Colunas obrigatórias ausentes no raw: {missing_cols}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("Não foi possível montar histórico válido a partir do arquivo raw.")

    return df


def _build_feature_window_from_close(close: float, feature_cols: list[str], seq_length: int):
    raw_df = load_raw_data(RAW_DATA_PATH)
    raw_df = _normalize_raw_data(raw_df)

    if len(raw_df) < 25:
        raise ValueError("Histórico insuficiente no arquivo raw para montar as features de inferência.")

    next_row = raw_df.iloc[-1].copy()
    next_row["date"] = pd.to_datetime(next_row["date"]) + pd.Timedelta(days=1)
    next_row["close"] = float(close)

    if "open" in raw_df.columns:
        next_row["open"] = float(close)
    if "high" in raw_df.columns:
        next_row["high"] = float(close)
    if "low" in raw_df.columns:
        next_row["low"] = float(close)
    if "volume" in raw_df.columns:
        next_row["volume"] = float(raw_df["volume"].iloc[-1])

    raw_df = pd.concat([raw_df, pd.DataFrame([next_row])], ignore_index=True)

    raw_df["return_1d"] = raw_df["close"].pct_change()
    raw_df["ma_5"] = raw_df["close"].rolling(5).mean()
    raw_df["ma_20"] = raw_df["close"].rolling(20).mean()
    raw_df["ma_5_ratio"] = raw_df["close"] / raw_df["ma_5"] - 1.0
    raw_df["ma_20_ratio"] = raw_df["close"] / raw_df["ma_20"] - 1.0
    raw_df["volatility_10"] = raw_df["return_1d"].rolling(10).std()

    if "volume" in raw_df.columns:
        vol_mean_20 = raw_df["volume"].rolling(20).mean()
        vol_std_20 = raw_df["volume"].rolling(20).std()
        raw_df["volume_zscore_20"] = (raw_df["volume"] - vol_mean_20) / vol_std_20.replace(0, pd.NA)
    else:
        raw_df["volume_zscore_20"] = 0.0

    raw_df = raw_df.replace([float("inf"), float("-inf")], pd.NA)
    raw_df = raw_df.dropna(subset=feature_cols).reset_index(drop=True)

    if len(raw_df) < seq_length:
        raise ValueError(
            f"Após gerar as features, não há linhas suficientes para inferência. "
            f"Necessário: {seq_length}, disponível: {len(raw_df)}."
        )

    window = raw_df[feature_cols].tail(seq_length).values
    last_close = float(raw_df["close"].iloc[-1])

    return window, last_close


def predict_from_close(close: float) -> dict:
    metadata = download_metadata_from_registry(
        model_name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    feature_cols = metadata["feature_cols"]
    seq_length = metadata["seq_length"]

    data, last_close = _build_feature_window_from_close(
        close=close,
        feature_cols=feature_cols,
        seq_length=seq_length,
    )

    prediction = predict_next_day(data, last_close=last_close)

    return {
        "model_name": REGISTERED_MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "seq_length": seq_length,
        "feature_cols": feature_cols,
        "last_close": prediction["last_close"],
        "predicted_return": prediction["predicted_return"],
        "predicted_price": prediction["predicted_price"],
    }