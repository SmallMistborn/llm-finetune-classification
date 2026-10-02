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
    mode: Literal["peft", "full"] = "full"
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=20_000,
        description="Если не задано — учим NUM_TRAIN_EPOCHS эпох (~2 на сбалансированном train)",
    )
    num_train_epochs: float = Field(default=2, gt=0, le=10)
    per_device_train_batch_size: int = Field(default=8, ge=1, le=32)
    learning_rate: float | None = Field(
        default=None,
        gt=0,
        description="По умолчанию 2e-5 для full, 1e-4 для peft",
    )
    output_dir: str | None = Field(
        default=None,
        description="Куда сохранить веса. По умолчанию SFT_OUTPUT_PEFT или SFT_OUTPUT",
    )

