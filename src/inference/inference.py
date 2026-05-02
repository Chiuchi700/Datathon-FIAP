import os

import numpy as np
from dotenv import load_dotenv

from src.data_loader import load_raw_data
from src.logger_config import setup_logger
from src.mlflow_utils import configure_mlflow_uris
from src.model_registry import download_metadata_from_registry, load_model_from_registry
from src.preprocessing import create_processed_data
from src.train import predict

load_dotenv()
logger = setup_logger("inference")

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "nike_lstm_forecaster")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "candidate")


def predict_next_day(data_last_n_days, last_close: float | None = None) -> dict:
    """
    Realiza a predição do próximo dia com base na última janela de dados.

    Parameters
    ----------
    data_last_n_days : array-like
        Estrutura no formato (window_size, n_features), contendo exatamente
        as mesmas features usadas no treinamento.
    last_close : float | None
        Último preço de fechamento conhecido. Quando informado, a função
        também calcula o preço previsto a partir do retorno previsto.

    Returns
    -------
    dict
        {
            "predicted_return": float,
            "last_close": float | None,
            "predicted_price": float | None
        }
    """
    configure_mlflow_uris(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        registry_uri=os.getenv("MLFLOW_REGISTRY_URI"),
    )

    metadata = download_metadata_from_registry(
        model_name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    scaler = metadata["scaler"]
    window_size = metadata["seq_length"]
    feature_cols = metadata["feature_cols"]
    expected_n_features = len(feature_cols)

    model = load_model_from_registry(
        model_name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    data_last_n_days = np.asarray(data_last_n_days, dtype=np.float32)

    if data_last_n_days.ndim == 1:
        if expected_n_features != 1:
            raise ValueError(
                "Entrada inválida: foi recebido um vetor 1D, mas o modelo espera "
                f"{expected_n_features} features por dia."
            )
        data_last_n_days = data_last_n_days.reshape(-1, 1)

    if data_last_n_days.ndim != 2:
        raise ValueError(
            "Entrada inválida: os dados devem estar no formato "
            "(window_size, n_features)."
        )

    received_window_size, received_n_features = data_last_n_days.shape

    if received_window_size != window_size:
        logger.error(
            "Quantidade de dias inválida | esperado=%s | recebido=%s",
            window_size,
            received_window_size,
        )
        raise ValueError(
            f"Esperado {window_size} dias para inferência, mas foram recebidos "
            f"{received_window_size}."
        )

    if received_n_features != expected_n_features:
        logger.error(
            "Quantidade de features inválida | esperado=%s | recebido=%s | features esperadas=%s",
            expected_n_features,
            received_n_features,
            feature_cols,
        )
        raise ValueError(
            f"Esperado {expected_n_features} features por dia, mas foram recebidas "
            f"{received_n_features}. Features esperadas: {feature_cols}"
        )

    scaled_input = scaler.transform(data_last_n_days)
    scaled_input = scaled_input.reshape(
        1,
        scaled_input.shape[0],
        scaled_input.shape[1],
    ).astype(np.float32)

    prediction_return = float(predict(model, scaled_input)[0])

    logger.info(
        "Retorno previsto calculado com sucesso: %.6f (%.4f%%)",
        prediction_return,
        prediction_return * 100,
    )

    result = {
        "predicted_return": prediction_return,
        "last_close": None,
        "predicted_price": None,
    }

    if last_close is not None:
        last_close = float(last_close)
        predicted_price = float(last_close * (1 + prediction_return))

        logger.info("Último fechamento conhecido: %.4f", last_close)
        logger.info("Preço previsto para o próximo dia: %.4f", predicted_price)

        result["last_close"] = last_close
        result["predicted_price"] = predicted_price

    return result


def main():
    logger.info("Iniciando inferência local do próximo dia")

    raw_df = load_raw_data("data/raw/nike_raw.csv")
    df = create_processed_data(raw_df)

    metadata = download_metadata_from_registry(
        model_name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    feature_cols = metadata["feature_cols"]
    seq_length = metadata["seq_length"]

    if len(df) < seq_length:
        raise ValueError(
            f"Dados insuficientes para inferência. O modelo exige pelo menos "
            f"{seq_length} linhas, mas o dataframe possui {len(df)}."
        )

    last_window = df[feature_cols].tail(seq_length).values.astype(np.float32)
    last_close = float(df["close"].iloc[-1])

    prediction = predict_next_day(last_window, last_close=last_close)

    logger.info("===== PREVISÃO DO PRÓXIMO DIA =====")
    logger.info("Último fechamento conhecido: %.4f", prediction["last_close"])
    logger.info(
        "Retorno previsto para o próximo dia: %.6f (%.4f%%)",
        prediction["predicted_return"],
        prediction["predicted_return"] * 100,
    )
    logger.info("Preço previsto para o próximo dia: %.4f", prediction["predicted_price"])


if __name__ == "__main__":
    main()