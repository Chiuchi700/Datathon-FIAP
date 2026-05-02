import logging
import os
import sys
from pathlib import Path


def setup_logger(name: str = "nike_forecast") -> logging.Logger:
    # No Docker, o LOG_DIR deve ser absoluto para garantir que caia no volume correto
    # Padrão: /opt/airflow/logs (conforme definido no seu docker-compose)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    default_log_dir = "/opt/airflow/logs"

    log_dir = Path(os.getenv("LOG_DIR", default_log_dir))

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Fallback para diretório local caso esteja rodando fora do container
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Evita duplicar handlers se o logger já estiver configurado
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler para o Console (Aparece no 'docker logs')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Handler para o Arquivo (Persiste no volume ./logs do host)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
