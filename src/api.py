from pathlib import Path
from threading import Thread
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.metrics import f1_score

from config import BASE_MODEL, SFT_OUTPUT, SFT_OUTPUT_PEFT
from get_device import get_device
from inference import load_model, predict
from model_service import model_service
from prepare_datasets import load_splits
from training import run_full_training, run_peft_training

app = FastAPI(
    title="LLM Fine-tune Classification API",
    description="Обучение и инференс модели классификации отзывов Kinopoisk",
    version="1.0.0",
)

train_status = {
    "status": "idle",  #возможные статусы обучения idle | running | completed | failed
    "mode": None,
    "output_dir": None,
    "error": None,
    "result": None,
}


class PredictRequest(BaseModel):
    movie_name: str = Field(..., examples=["Блеф (1976)"])
    review: str = Field(..., examples=["Отличный фильм, очень смешно!"])
    model_path: str | None = Field(
        default=None,
        description="Путь к адаптеру или чекпоинту. По умолчанию SFT_OUTPUT_PEFT из .env",
    )
    max_new_tokens: int = Field(default=8, ge=1, le=32)


class EvaluateRequest(BaseModel):
    model_path: str | None = None
    limit: int = Field(default=20, ge=1, le=500)
    max_new_tokens: int = Field(default=8, ge=1, le=32)


class TrainRequest(BaseModel):
    mode: Literal["peft", "full"] = "peft"
    max_steps: int = Field(default=50, ge=1, le=10_000)
    per_device_train_batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=1e-4, gt=0)
    output_dir: str | None = Field(
        default=None,
        description="Куда сохранить веса. По умолчанию SFT_OUTPUT_PEFT или SFT_OUTPUT",
    )


def default_output_dir(mode: str) -> str:
    if mode == "peft":
        return str(SFT_OUTPUT_PEFT)
    return str(SFT_OUTPUT)


def ensure_not_training() -> None:
    if train_status["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Идёт обучение. Дождитесь завершения или проверьте /train/status",
        )


def run_training_job(mode: str, params: dict) -> None:
    """Запускается в отдельном потоке."""
    global train_status
    try:
        if mode == "peft":
            result = run_peft_training(**params)
        elif mode == "full":
            result = run_full_training(**params)
        else:
            raise ValueError(f"Неизвестный mode: {mode}")

        train_status = {
            "status": "completed",
            "mode": mode,
            "output_dir": result["output_dir"],
            "error": None,
            "result": result,
        }
        model_service.unload()
    except Exception as exc:
        train_status = {
            "status": "failed",
            "mode": mode,
            "output_dir": params.get("output_dir"),
            "error": str(exc),
            "result": None,
        }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "base_model": BASE_MODEL,
        "device": get_device(),
        "model": model_service.status(),
        "training": train_status,
    }


@app.get("/model/status")
def model_status() -> dict:
    return model_service.status()


@app.post("/model/reload")
def reload_model(model_path: str | None = None) -> dict:
    model_service.unload()
    model_service.load(model_path)
    return model_service.status()


@app.post("/predict")
def predict_review(request: PredictRequest) -> dict:
    ensure_not_training()
    return model_service.predict(
        movie_name=request.movie_name,
        review=request.review,
        model_path=request.model_path,
        max_new_tokens=request.max_new_tokens,
    )


@app.post("/evaluate")
def evaluate_model(request: EvaluateRequest) -> dict:
    ensure_not_training()
    model_path = request.model_path or str(SFT_OUTPUT_PEFT)
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Модель не найдена: {model_path}")

    device = get_device()
    model, tokenizer, model_type = load_model(model_path, device)
    _train, _val, test = load_splits()
    if request.limit:
        test = test.select(range(min(request.limit, len(test))))
    labels = ["Bad", "Good", "Neutral"]
    y_true, y_pred = [], []
    for ex in test:
        pred = predict(
            model,
            tokenizer,
            ex["movie_name"],
            ex["content"],
            device,
            request.max_new_tokens,
        )
        y_true.append(ex["grade3"])
        y_pred.append(pred)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    n = len(y_true)
    return {
        "model": model_path,
        "model_type": model_type,
        "split": "test",
        "accuracy": correct / n if n else 0.0,
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "total": n,
        "correct": correct,
    }


@app.post("/train", status_code=202)
def start_training(request: TrainRequest) -> dict:
    if train_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Обучение уже запущено")

    model_service.unload()
    output_dir = request.output_dir or default_output_dir(request.mode)
    params = {
        "max_steps": request.max_steps,
        "per_device_train_batch_size": request.per_device_train_batch_size,
        "learning_rate": request.learning_rate,
        "output_dir": output_dir,
    }

    train_status.clear()
    train_status.update(
        {
            "status": "running",
            "mode": request.mode,
            "output_dir": output_dir,
            "error": None,
            "result": None,
        }
    )

    thread = Thread(
        target=run_training_job,
        args=(request.mode, params),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Обучение запущено",
        "mode": request.mode,
        "output_dir": output_dir,
        "training": train_status,
    }


@app.get("/train/status")
def training_status() -> dict:
    return train_status
