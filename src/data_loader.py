from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)



def download_nike_data(ticker: str = "NKE", months: int = 60) -> tuple[pd.DataFrame, Path]:
    today = datetime.today()
    start_date = today - relativedelta(months=months)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )

    df.reset_index(inplace=True)
    raw_path = RAW_DIR / "nike_raw.csv"
    df.to_csv(raw_path, index=False)

    return df, raw_path



def save_raw_data(df: pd.DataFrame, path: str | Path = RAW_DIR / "nike_raw.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path



def load_raw_data(path: str | Path = "data/raw/nike_raw.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])
