from typing import List

from pydantic import BaseModel, Field


class FeatureRow(BaseModel):
    close: float = Field(..., description="Preço de fechamento")
    return_1d: float = Field(..., description="Retorno diário")
    ma_5_ratio: float = Field(..., description="Razão entre close e média móvel de 5 dias")
    ma_20_ratio: float = Field(..., description="Razão entre close e média móvel de 20 dias")
    volatility_10: float = Field(..., description="Volatilidade de 10 dias")
    volume_zscore_20: float = Field(..., description="Z-score do volume em 20 dias")


class PredictRequest(BaseModel):
    close: float = Field(..., description="Último preço de fechamento conhecido")


class PredictResponse(BaseModel):
    model_name: str
    model_alias: str
    seq_length: int
    feature_cols: list[str]
    last_close: float
    predicted_return: float
    predicted_price: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_alias: str
    seq_length: int
    feature_cols: list[str]


class ExplainRequest(BaseModel):
    close: float = Field(..., description="Ultimo preco de fechamento conhecido")


class ExplainResponse(BaseModel):
    explanation: str
    close: float
    predicted_price: float
    predicted_return: float


class ChatRequest(BaseModel):
    question: str = Field(..., description="Pergunta sobre o modelo ou projeto")


class ChatResponse(BaseModel):
    answer: str
    question: str