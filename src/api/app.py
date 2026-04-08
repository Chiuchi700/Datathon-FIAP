from fastapi import FastAPI, HTTPException

from src.api.schema import ModelInfoResponse, PredictRequest, PredictResponse
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