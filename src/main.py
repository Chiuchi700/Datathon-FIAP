from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Necessário para rodar em containers sem interface gráfica
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from src.logger_config import setup_logger
from src.mlflow_utils import setup_mlflow
from src.model_registry import register_run_model, set_model_alias
from src.preprocessing import load_processed_data, prepare_sequences
from src.train import build_lstm_model, predict, train_model

load_dotenv()
logger = setup_logger("main")

# Configuração de Caminhos Globais para Docker
PROJECT_ROOT = Path("/opt/airflow/project")
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    """Carrega hiperparâmetros do arquivo YAML."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> Path:
    """Garante que a pasta pai de um arquivo exista."""
    path = Path(path)
    # Se o path for relativo, ancora no PROJECT_ROOT
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def direction_accuracy(y_true_prices, y_pred_prices):
    """Calcula a acurácia da direção (subida/descida)."""
    real_dir = np.sign(np.diff(y_true_prices.flatten()))
    pred_dir = np.sign(np.diff(y_pred_prices.flatten()))
    if len(real_dir) == 0:
        return np.nan
    return (real_dir == pred_dir).mean()


def predict_next_trading_day(model, df, artifacts):
    """Realiza a previsão para o próximo dia útil."""
    feature_cols = artifacts["feature_cols"]
    seq_length = artifacts["seq_length"]
    scaler = artifacts["scaler"]

    last_window = df[feature_cols].tail(seq_length).copy()
    last_close = df["close"].iloc[-1]

    last_window_scaled = scaler.transform(last_window)
    last_window_scaled = np.expand_dims(last_window_scaled, axis=0).astype(np.float32)

    next_day_return_pred = predict(model, last_window_scaled)[0]
    next_day_price_pred = last_close * (1 + next_day_return_pred)

    return {
        "last_close": float(last_close),
        "predicted_return": float(next_day_return_pred),
        "predicted_price": float(next_day_price_pred),
    }


# --- Funções de Logging e Plotagem ---


def log_model_summary(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    logger.info("Arquitetura do modelo:\n%s", model)
    logger.info(
        "Parâmetros totais: %s | treináveis: %s", total_params, trainable_params
    )


def save_training_plot(history, output_path: Path):
    output_path = ensure_parent(output_path)
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss de treino vs validação")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_prediction_plot(y_test_prices, pred_prices, naive_prices, output_path: Path):
    output_path = ensure_parent(output_path)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_prices, color="black", label="Real")
    plt.plot(pred_prices, color="green", label="Previsto")
    plt.plot(naive_prices, color="orange", linestyle="--", label="Baseline ingênuo")
    plt.legend()
    plt.title("Previsão vs Real")
    plt.xlabel("Amostras (Teste)")
    plt.ylabel("Preço")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_history_csv(history, output_path: Path):
    output_path = ensure_parent(output_path)
    df_history = pd.DataFrame(
        {
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }
    )
    df_history.to_csv(output_path, index=False)


def save_predictions_csv(
    test_dates, y_test_prices, pred_prices, naive_prices, output_path: Path
):
    output_path = ensure_parent(output_path)
    predictions_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "real_price": y_test_prices.flatten(),
            "pred_price": pred_prices.flatten(),
            "naive_price": naive_prices.flatten(),
        }
    )
    predictions_df.to_csv(output_path, index=False)


def save_metrics_json(metrics: dict, output_path: Path):
    output_path = ensure_parent(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_local_model_artifacts(
    model, artifacts: dict, model_path: Path, metadata_path: Path
):
    model_path = ensure_parent(model_path)
    metadata_path = ensure_parent(metadata_path)
    torch.save(model, model_path)
    joblib.dump(artifacts, metadata_path)


def log_preprocessing_metadata(artifacts: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        metadata_path = Path(tmp_dir) / "model_metadata.pkl"
        joblib.dump(artifacts, metadata_path)
        mlflow.log_artifact(str(metadata_path), artifact_path="preprocessing")


# --- Pipeline Principal ---


def main():
    try:
        params = load_params()

        data_cfg = params["data"]
        train_cfg = params["train"]
        output_cfg = params["outputs"]
        mlflow_cfg = params["mlflow"]

        # Configuração de caminhos usando o root do projeto
        processed_path = data_cfg["processed_path"]
        model_path = Path(output_cfg["model_path"])
        metadata_path = Path(output_cfg["metadata_path"])
        metrics_path = Path(output_cfg["metrics_path"])
        history_path = Path(output_cfg["history_path"])
        predictions_path = Path(output_cfg["predictions_path"])
        training_plot_path = Path(output_cfg["training_plot_path"])
        prediction_plot_path = Path(output_cfg["prediction_plot_path"])

        logger.info("Iniciando pipeline de treino LSTM (PyTorch)")
        df = load_processed_data(processed_path)

        # Prioriza o params.yaml para manter consistência no Docker
        experiment_id = setup_mlflow(
            experiment_name=mlflow_cfg["experiment_name"],
            tracking_uri=mlflow_cfg["tracking_uri"],
            registry_uri=mlflow_cfg["registry_uri"],
            artifact_location=mlflow_cfg["artifact_root"],
        )
        logger.info("MLflow configurado | experiment_id=%s", experiment_id)

        with mlflow.start_run(run_name="nike_lstm_pytorch"):
            mlflow.set_tag("model_framework", "pytorch")
            mlflow.set_tag("project", "datathon")
            mlflow.set_tag("asset", data_cfg["ticker"])

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                prev_close_train,
                prev_close_val,
                prev_close_test,
                dates_train,
                dates_val,
                dates_test,
                artifacts,
            ) = prepare_sequences(
                df,
                seq_length=train_cfg["seq_length"],
                train_split=train_cfg["train_split"],
                val_split=train_cfg["val_split"],
            )

            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            log_model_summary(model)

            # Log de Parâmetros
            mlflow.log_param("ticker", data_cfg["ticker"])
            for key, value in train_cfg.items():
                mlflow.log_param(key, value)

            # Treinamento
            model, history = train_model(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=train_cfg["epochs"],
                batch_size=train_cfg["batch_size"],
                learning_rate=train_cfg["learning_rate"],
                patience=train_cfg["patience"],
            )

            # Avaliação
            pred_returns = predict(model, X_test).reshape(-1, 1)
            pred_prices = prev_close_test.reshape(-1, 1) * (1 + pred_returns)
            y_test_prices = prev_close_test.reshape(-1, 1) * (1 + y_test.reshape(-1, 1))
            naive_prices = prev_close_test.reshape(-1, 1)

            rmse = np.sqrt(mean_squared_error(y_test_prices, pred_prices))
            mae = mean_absolute_error(y_test_prices, pred_prices)
            dir_acc = direction_accuracy(y_test_prices, pred_prices)
            next_pred = predict_next_trading_day(model, df, artifacts)

            metrics = {
                "rmse": float(rmse),
                "mae": float(mae),
                "direction_accuracy": float(dir_acc) if not np.isnan(dir_acc) else 0.0,
                "predicted_price_next_day": float(next_pred["predicted_price"]),
            }

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Salvamento e Artefatos
            save_local_model_artifacts(model, artifacts, model_path, metadata_path)
            save_metrics_json(metrics, metrics_path)
            save_history_csv(history, history_path)
            save_predictions_csv(
                dates_test, y_test_prices, pred_prices, naive_prices, predictions_path
            )
            save_training_plot(history, training_plot_path)
            save_prediction_plot(
                y_test_prices, pred_prices, naive_prices, prediction_plot_path
            )

            # Log MLflow Artefatos
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")
            mlflow.log_artifact(str(prediction_plot_path), artifact_path="plots")
            log_preprocessing_metadata(artifacts)

            # Log do Modelo com Assinatura
            input_example = X_train[:1]
            signature = mlflow.models.infer_signature(
                input_example, predict(model, input_example)
            )
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=mlflow_cfg["model_artifact_name"],
                signature=signature,
            )

            # Registro de Modelo
            if mlflow_cfg["register_model"]:
                run_id = mlflow.active_run().info.run_id
                model_version = register_run_model(
                    run_id=run_id,
                    model_name=mlflow_cfg["registered_model_name"],
                    model_artifact_name=mlflow_cfg["model_artifact_name"],
                )
                set_model_alias(
                    model_name=mlflow_cfg["registered_model_name"],
                    version=model_version.version,
                    alias=mlflow_cfg["model_alias"],
                )

            logger.info("Pipeline finalizado com sucesso")

    except Exception:
        logger.exception("Erro fatal no pipeline")
        raise


if __name__ == "__main__":
    main()
