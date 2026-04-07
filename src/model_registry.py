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

DEFAULT_MODEL_NAME = "nike_lstm_forecaster"
DEFAULT_MODEL_ARTIFACT_NAME = "model"
DEFAULT_METADATA_ARTIFACT_PATH = "preprocessing/model_metadata.pkl"



def _get_client() -> MlflowClient:
    configure_mlflow_uris(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        registry_uri=os.getenv("MLFLOW_REGISTRY_URI"),
    )
    return MlflowClient()



def ensure_registered_model(model_name: str) -> None:
    client = _get_client()

    try:
        client.get_registered_model(model_name)
        logger.info("Registered Model '%s' já existe.", model_name)
        return
    except MlflowException:
        pass

    client.create_registered_model(model_name)
    logger.info("Registered Model '%s' criado com sucesso.", model_name)



def wait_until_model_version_is_ready(model_name: str, version: str | int, timeout_s: int = 60):
    client = _get_client()
    version = str(version)
    start = time.time()

    while time.time() - start <= timeout_s:
        model_version = client.get_model_version(name=model_name, version=version)
        status = ModelVersionStatus.from_string(model_version.status)

        if status == ModelVersionStatus.READY:
            logger.info(
                "Model version pronta | name=%s | version=%s | status=%s",
                model_name,
                version,
                model_version.status,
            )
            return model_version

        logger.info(
            "Aguardando model version ficar pronta | name=%s | version=%s | status=%s",
            model_name,
            version,
            model_version.status,
        )
        time.sleep(2)

    raise TimeoutError(
        f"Timeout aguardando model version READY | model_name={model_name} | version={version}"
    )



def register_run_model(
    run_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model_artifact_name: str = DEFAULT_MODEL_ARTIFACT_NAME,
):
    ensure_registered_model(model_name)

    model_uri = f"runs:/{run_id}/{model_artifact_name}"
    logger.info("Registrando modelo a partir de: %s", model_uri)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    ready_model_version = wait_until_model_version_is_ready(
        model_name=model_version.name,
        version=model_version.version,
    )

    logger.info(
        "Modelo registrado com sucesso | name=%s | version=%s",
        ready_model_version.name,
        ready_model_version.version,
    )
    return ready_model_version



def set_model_alias(model_name: str, version: str | int, alias: str = "champion") -> None:
    client = _get_client()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(version),
    )
    logger.info(
        "Alias '%s' definido para %s versão %s",
        alias,
        model_name,
        version,
    )



def get_model_version_by_alias(model_name: str = DEFAULT_MODEL_NAME, alias: str = "champion"):
    client = _get_client()
    model_version = client.get_model_version_by_alias(model_name, alias)
    logger.info(
        "Versão encontrada por alias | model_name=%s | alias=%s | version=%s | run_id=%s",
        model_name,
        alias,
        model_version.version,
        model_version.run_id,
    )
    return model_version



def build_registry_model_uri(model_name: str = DEFAULT_MODEL_NAME, alias: str = "champion") -> str:
    return f"models:/{model_name}@{alias}"



def load_model_from_registry(model_name: str = DEFAULT_MODEL_NAME, alias: str = "champion"):
    configure_mlflow_uris(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        registry_uri=os.getenv("MLFLOW_REGISTRY_URI"),
    )
    model_uri = build_registry_model_uri(model_name=model_name, alias=alias)
    logger.info("Carregando modelo do registry: %s", model_uri)
    return mlflow.pytorch.load_model(model_uri)



def download_metadata_from_registry(
    model_name: str = DEFAULT_MODEL_NAME,
    alias: str = "champion",
    metadata_artifact_path: str = DEFAULT_METADATA_ARTIFACT_PATH,
):
    configure_mlflow_uris(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        registry_uri=os.getenv("MLFLOW_REGISTRY_URI"),
    )
    model_version = get_model_version_by_alias(model_name=model_name, alias=alias)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id,
        artifact_path=metadata_artifact_path,
    )
    logger.info("Metadata baixada localmente em: %s", local_path)
    return joblib.load(Path(local_path))
