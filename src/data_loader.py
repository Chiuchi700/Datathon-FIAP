from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_nike_data(ticker="NKE", months=60):
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


def load_raw_data(path="data/raw/nike_raw.csv"):
    return pd.read_csv(path, parse_dates=["Date"])
