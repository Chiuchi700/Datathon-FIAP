import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

# Default vale para o container Airflow; pode ser sobrescrito por env var (testes locais, dev fora do container).
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_RAW_PATH = RAW_DIR / "nike_raw.csv"


def download_nike_data(ticker="NKE", months=60):
    """
    Faz o download dos dados via yfinance e salva no diretório raw.
    """
    today = datetime.today()
    start_date = today - relativedelta(months=months)

    # Download usando auto_adjust para preços reais
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )

    # Reset do index para transformar a Date em coluna
    df.reset_index(inplace=True)

    # Salva no caminho absoluto do container
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_RAW_PATH, index=False)

    return df, DEFAULT_RAW_PATH


def load_raw_data(path=None):
    """
    Carrega os dados crus. Se nenhum path for passado, usa o padrão do container.
    """
    if path is None:
        path = DEFAULT_RAW_PATH

    return pd.read_csv(path, parse_dates=["Date"])


def save_raw_data(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Salva um DataFrame garantindo que a estrutura de pastas exista.
    """
    path = Path(path)
    # Se o path for relativo (ex: 'data/raw/x.csv'), concatena com o root do projeto
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
