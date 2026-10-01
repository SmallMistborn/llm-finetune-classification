from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import BASE_MODEL
from model_path import ModelNotFoundError
from model_service import model_service
from schemas import EvaluateRequest, PredictRequest, TrainRequest
from training_job import TrainingAlreadyRunningError, training_service

app = FastAPI(
    title="LLM Fine-tune Classification API",
    description="Обучение и инференс модели классификации отзывов Kinopoisk",
    version="1.0.0",
)


@app.exception_handler(ModelNotFoundError)
def model_not_found_handler(_request: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(TrainingAlreadyRunningError)
def training_busy_handler(_request: Request, exc: TrainingAlreadyRunningError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "base_model": BASE_MODEL,
        "model": model_service.status(),
        "training": training_service.status(),
    }


@app.get("/model/status")
def model_status() -> dict:
    return model_service.status()


@app.post("/model/reload")
def reload_model(model_path: str | None = None) -> dict:
    training_service.require_idle()
    return model_service.reload(model_path)


@app.post("/predict")
def predict_review(request: PredictRequest) -> dict:
    training_service.require_idle()
    return model_service.predict(
        movie_name=request.movie_name,
        review=request.review,
        model_path=request.model_path,
    )


@app.post("/evaluate")
def evaluate_model(request: EvaluateRequest) -> dict:
    training_service.require_idle()
    return model_service.evaluate(
        model_path=request.model_path,
        limit=request.limit,
    )


@app.post("/train", status_code=202)
def start_training(request: TrainRequest) -> dict:
    return training_service.start(
        mode=request.mode,
        max_steps=request.max_steps,
        per_device_train_batch_size=request.per_device_train_batch_size,
        learning_rate=request.learning_rate,
        output_dir=request.output_dir,
    )


@app.get("/train/status")
def training_status() -> dict:
    return training_service.status()
