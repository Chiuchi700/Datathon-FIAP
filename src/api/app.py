from fastapi import FastAPI, HTTPException

from src.api.llm_service import explain_prediction
from src.api.rag_service import chat_about_model
from src.api.schema import (
    ChatRequest,
    ChatResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.service import get_model_info, predict_from_close

app = FastAPI(
    title="Nike Forecast API",
    description="API de inferência do modelo LSTM registrado no MLflow",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Nike Forecast API está no ar"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return get_model_info()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        return predict_from_close(request.close)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    try:
        prediction = predict_from_close(request.close)
        explanation = explain_prediction(
            close=request.close,
            predicted_price=prediction["predicted_price"],
            predicted_return=prediction["predicted_return"],
        )
        return ExplainResponse(
            explanation=explanation,
            close=request.close,
            predicted_price=prediction["predicted_price"],
            predicted_return=prediction["predicted_return"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = chat_about_model(request.question)
        return ChatResponse(answer=answer, question=request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))