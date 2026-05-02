import os
import time
from pathlib import Path
import joblib
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException

from src.logger_config import setup_logger
from src.mlflow_utils import configure_mlflow_uris

logger = setup_logger("model_registry")

# Padrões consistentes com o params.yaml
DEFAULT_MODEL_NAME = "nike_lstm_forecaster"
DEFAULT_MODEL_ARTIFACT_NAME = "model"
DEFAULT_METADATA_ARTIFACT_PATH = "preprocessing/model_metadata.pkl"


def _get_client() -> MlflowClient:
    """
    Retorna o cliente MLflow garantindo que as URIs de tracking/registry
    estejam configuradas conforme o ambiente (Docker ou Local).
    """
    # Se já houver uma URI setada no mlflow global, o MlflowClient a usará.
    # Caso contrário, tentamos pegar do ambiente ou usamos o padrão.
    if not mlflow.get_tracking_uri():
        configure_mlflow_uris()
    return MlflowClient()


def ensure_registered_model(model_name: str) -> None:
    """Garante que o container do modelo exista no Registry."""
    client = _get_client()
    try:
        client.get_registered_model(model_name)
        logger.info("Modelo registrado '%s' já existe.", model_name)
        return
    except MlflowException:
        pass

    client.create_registered_model(model_name)
    logger.info("Modelo registrado '%s' criado com sucesso.", model_name)


def wait_until_model_version_is_ready(
    model_name: str, version: str | int, timeout_s: int = 60
):
    """
    Bloqueia a execução até que o MLflow processe o modelo e o deixe pronto (READY).
    Crucial para evitar erros de 'File Not Found' no Registry.
    """
    client = _get_client()
    version = str(version)
    start = time.time()

    while time.time() - start <= timeout_s:
        model_version = client.get_model_version(name=model_name, version=version)
        status = ModelVersionStatus.from_string(model_version.status)

        if status == ModelVersionStatus.READY:
            logger.info("Versão %s pronta!", version)
            return model_version

        logger.info("Aguardando versão %s (Status: %s)...", version, status)
        time.sleep(2)

    raise TimeoutError(f"Timeout aguardando modelo {model_name} v{version}")


def register_run_model(
    run_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model_artifact_name: str = DEFAULT_MODEL_ARTIFACT_NAME,
):
    """Registra um modelo treinado a partir de uma Run ID específica."""
    ensure_registered_model(model_name)

    model_uri = f"runs:/{run_id}/{model_artifact_name}"
    logger.info("Registrando modelo no Registry a partir de: %s", model_uri)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    return wait_until_model_version_is_ready(
        model_name=model_version.name,
        version=model_version.version,
    )


def set_model_alias(
    model_name: str, version: str | int, alias: str = "champion"
) -> None:
    """Define um alias (ex: 'champion') para uma versão específica."""
    client = _get_client()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(version),
    )
    logger.info(
        "Alias '%s' aplicado à versão %s do modelo %s.", alias, version, model_name
    )


def get_model_version_by_alias(
    model_name: str = DEFAULT_MODEL_NAME, alias: str = "champion"
):
    """Recupera metadados da versão que possui o alias informado."""
    client = _get_client()
    return client.get_model_version_by_alias(model_name, alias)


def load_model_from_registry(
    model_name: str = DEFAULT_MODEL_NAME, alias: str = "champion"
):
    """
    Carrega o modelo PyTorch diretamente do Registry usando o alias.
    Utilizado pela FastAPI.
    """
    model_uri = f"models:/{model_name}@{alias}"
    logger.info("Carregando modelo via Registry: %s", model_uri)
    return mlflow.pytorch.load_model(model_uri)


def download_metadata_from_registry(
    model_name: str = DEFAULT_MODEL_NAME,
    alias: str = "champion",
    metadata_artifact_path: str = DEFAULT_METADATA_ARTIFACT_PATH,
):
    """
    Baixa os arquivos de pré-processamento (Scaler, etc) vinculados ao modelo champion.
    """
    model_version = get_model_version_by_alias(model_name=model_name, alias=alias)

    # O MLflow baixa para uma pasta temporária e retorna o caminho
    local_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id,
        artifact_path=metadata_artifact_path,
    )
    logger.info("Metadata baixada em: %s", local_path)
    return joblib.load(Path(local_path))
