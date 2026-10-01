from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    movie_name: str = Field(..., examples=["Блеф (1976)"])
    review: str = Field(..., examples=["Отличный фильм, очень смешно!"])
    model_path: str | None = Field(
        default=None,
        description="Путь к адаптеру или чекпоинту. По умолчанию SFT_OUTPUT_PEFT из .env",
    )


class EvaluateRequest(BaseModel):
    model_path: str | None = None
    limit: int = Field(default=20, ge=1, le=500)


class TrainRequest(BaseModel):
    mode: Literal["peft", "full"] = "peft"
    max_steps: int = Field(default=1000, ge=1, le=10_000)
    per_device_train_batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=1e-4, gt=0)
    output_dir: str | None = Field(
        default=None,
        description="Куда сохранить веса. По умолчанию SFT_OUTPUT_PEFT или SFT_OUTPUT",
    )

