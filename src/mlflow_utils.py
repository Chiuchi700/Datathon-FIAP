import os
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from src.logger_config import setup_logger

logger = setup_logger("mlflow_utils")

# Default vale para o container Airflow; pode ser sobrescrito por env var (testes locais, dev fora do container).
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
DEFAULT_DB_PATH = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
DEFAULT_ARTIFACT_PATH = f"file:{PROJECT_ROOT}/mlartifacts"

# URIs Padrão
DEFAULT_CLIENT_TRACKING_URI = DEFAULT_DB_PATH
DEFAULT_ARTIFACT_ROOT = DEFAULT_ARTIFACT_PATH
# Nome do serviço definido no seu docker-compose.yml
DEFAULT_SERVER_TRACKING_URI = "http://mlflow-ui:5000"


def _normalize_artifact_location(artifact_location: str | None) -> str:
    """
    Garante que o local dos artefatos seja um URI válido e absoluto.
    """
    if not artifact_location:
        return DEFAULT_ARTIFACT_ROOT

    if ":" in artifact_location:
        return artifact_location

    # Se for um caminho relativo, ancora no volume do Docker
    if not artifact_location.startswith("/"):
        return f"file:{PROJECT_ROOT}/{artifact_location}"

    return Path(artifact_location).resolve().as_uri()


def _is_http_uri(uri: str) -> bool:
    return uri.startswith("http://") or uri.startswith("https://")


def configure_mlflow_uris(
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> tuple[str, str]:
    """Configura onde o MLflow deve salvar os logs e o registro de modelos."""
    tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", DEFAULT_CLIENT_TRACKING_URI
    )
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
    """Verifica se o experimento existe ou cria um novo com o local de artefatos correto."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None:
        logger.info(
            "Experimento já existente | nome=%s | experiment_id=%s",
            experiment_name,
            experiment.experiment_id,
        )
        mlflow.set_experiment(experiment_name)
        return experiment.experiment_id

    tracking_uri = mlflow.get_tracking_uri()
    normalized_artifact_location = _normalize_artifact_location(artifact_location)

    # Se for HTTP (servidor), o servidor gerencia o local dos artefatos
    if _is_http_uri(tracking_uri):
        experiment_id = client.create_experiment(name=experiment_name)
        logger.info("Experimento criado via servidor HTTP | ID: %s", experiment_id)
    else:
        # Se for SQLite local (nosso caso padrão no Airflow), definimos o path dos artefatos
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=normalized_artifact_location,
        )
        logger.info(
            "Experimento criado localmente | ID: %s | Artifacts: %s",
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
    Ponto de entrada principal para configurar o MLflow no pipeline.
    """
    tracking_uri, registry_uri = configure_mlflow_uris(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )

    # Prioriza: Parâmetro da função -> Variável de ambiente -> Valor padrão absoluto
    final_artifact_location = artifact_location or os.getenv(
        "MLFLOW_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT
    )

    experiment_id = ensure_experiment(
        experiment_name=experiment_name,
        artifact_location=final_artifact_location,
    )

    logger.info("MLflow pronto para logar dados.")
    return experiment_id
