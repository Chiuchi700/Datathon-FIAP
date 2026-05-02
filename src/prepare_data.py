from __future__ import annotations
from pathlib import Path
import yaml
import os

# Imports internos com o prefixo do pacote src
from src.data_loader import download_nike_data, save_raw_data
from src.logger_config import setup_logger
from src.preprocessing import (
    create_processed_data,
    save_processed_data,
)

logger = setup_logger("prepare_data")

# Default vale para o container Airflow; pode ser sobrescrito por env var (testes locais, dev fora do container).
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    """
    Carrega o arquivo de parâmetros.
    Tenta o caminho absoluto (Docker) e cai para relativo se não encontrar.
    """
    path = Path(params_path)
    if not path.exists():
        path = Path("params.yaml")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    try:
        # 1. Carregar Configurações
        params = load_params()
        data_cfg = params["data"]

        ticker = data_cfg["ticker"]
        months = data_cfg["months"]

        # Garantir caminhos absolutos para evitar que o Airflow grave em locais errados
        raw_path = Path(data_cfg["raw_path"])
        if not raw_path.is_absolute():
            raw_path = PROJECT_ROOT / raw_path

        processed_path = Path(data_cfg["processed_path"])
        if not processed_path.is_absolute():
            processed_path = PROJECT_ROOT / processed_path

        # 2. Download dos Dados
        logger.info(
            "Iniciando ingestão de dados | Ticker: %s | Janela: %s meses",
            ticker,
            months,
        )
        raw_df, _ = download_nike_data(ticker=ticker, months=months)

        # 3. Salvamento do Raw
        saved_raw_path = save_raw_data(raw_df, raw_path)
        logger.info("Dados brutos armazenados em: %s", saved_raw_path)

        # 4. Pré-processamento
        logger.info("Iniciando transformações de pré-processamento...")
        processed_df = create_processed_data(raw_df)

        # 5. Salvamento do Processed
        saved_processed_path = save_processed_data(processed_df, processed_path)
        logger.info("Dados processados armazenados em: %s", saved_processed_path)
        logger.info("Pipeline de dados concluído | Shape final: %s", processed_df.shape)

    except Exception as e:
        logger.error("Falha na execução do prepare_data: %s", str(e))
        raise


if __name__ == "__main__":
    main()
