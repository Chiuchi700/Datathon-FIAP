from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.logger_config import setup_logger

logger = setup_logger("mlflow_utils")

DEFAULT_LOCAL_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_SERVER_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_CLIENT_TRACKING_URI = DEFAULT_LOCAL_TRACKING_URI
DEFAULT_ARTIFACT_ROOT = "file:./mlartifacts"



def _normalize_artifact_location(artifact_location: str | None) -> str:
    if not artifact_location:
        artifact_dir = Path("mlartifacts").resolve()
        return artifact_dir.as_uri()

    if ":" in artifact_location:
        return artifact_location

    return Path(artifact_location).resolve().as_uri()



def _is_http_uri(uri: str) -> bool:
    return uri.startswith("http://") or uri.startswith("https://")



def configure_mlflow_uris(
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> tuple[str, str]:
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_CLIENT_TRACKING_URI)
    registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    logger.info("MLflow tracking URI configurada: %s", tracking_uri)
    logger.info("MLflow registry URI configurada: %s", registry_uri)
    return tracking_uri, registry_uri



def ensure_experiment(
    experiment_name: str,
    artifact_location: str | None = None,
) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None:
        logger.info(
            "Experimento já existente | nome=%s | experiment_id=%s | artifact_location=%s",
            experiment_name,
            experiment.experiment_id,
            experiment.artifact_location,
        )
        mlflow.set_experiment(experiment_name)
        return experiment.experiment_id

    tracking_uri = mlflow.get_tracking_uri()
    normalized_artifact_location = _normalize_artifact_location(artifact_location)

    if _is_http_uri(tracking_uri):
        experiment_id = client.create_experiment(name=experiment_name)
        logger.info(
            "Experimento criado via servidor | nome=%s | experiment_id=%s",
            experiment_name,
            experiment_id,
        )
    else:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=normalized_artifact_location,
        )
        logger.info(
            "Experimento criado em modo local | nome=%s | experiment_id=%s | artifact_location=%s",
            experiment_name,
            experiment_id,
            normalized_artifact_location,
        )

    mlflow.set_experiment(experiment_name)
    return experiment_id



def setup_mlflow(
    experiment_name: str = "nike_lstm_forecasting",
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    artifact_location: str | None = None,
) -> str:
    """
    Configura tracking e registry para apontarem para a MESMA instância do MLflow.

    Comportamento padrão deste projeto:
      - o script Python grava direto no backend local sqlite:///mlflow.db
      - a UI pode ser aberta separadamente, apontando para o mesmo backend

    Assim, `python main.py` treina normalmente sem depender de um servidor HTTP
    já estar em execução. Se quiser usar um servidor MLflow remoto, basta trocar
    MLFLOW_TRACKING_URI e MLFLOW_REGISTRY_URI para a URL HTTP correspondente.
    """
    tracking_uri, registry_uri = configure_mlflow_uris(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )

    experiment_id = ensure_experiment(
        experiment_name=experiment_name,
        artifact_location=artifact_location or os.getenv("MLFLOW_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT),
    )

    logger.info(
        "MLflow pronto | experiment_name=%s | experiment_id=%s | tracking_uri=%s | registry_uri=%s",
        experiment_name,
        experiment_id,
        tracking_uri,
        registry_uri,
    )
    return experiment_id
