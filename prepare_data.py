from __future__ import annotations

from pathlib import Path

import yaml

from data_loader import download_nike_data, save_raw_data
from logger_config import setup_logger
from preprocessing import create_processed_data, save_processed_data

logger = setup_logger("prepare_data")


def load_params(params_path: str | Path = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    data_cfg = params["data"]

    ticker = data_cfg["ticker"]
    months = data_cfg["months"]
    raw_path = Path(data_cfg["raw_path"])
    processed_path = Path(data_cfg["processed_path"])

    logger.info("Baixando dados brutos | ticker=%s | months=%s", ticker, months)
    raw_df, _ = download_nike_data(ticker=ticker, months=months)
    saved_raw_path = save_raw_data(raw_df, raw_path)
    logger.info("Arquivo raw salvo em: %s", saved_raw_path)

    processed_df = create_processed_data(raw_df)
    saved_processed_path = save_processed_data(processed_df, processed_path)
    logger.info("Arquivo processed salvo em: %s", saved_processed_path)
    logger.info("Shape do processed: %s", processed_df.shape)


if __name__ == "__main__":
    main()
