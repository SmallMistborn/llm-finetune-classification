import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


BASE_MODEL = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
SFT_OUTPUT = _resolve_path(os.getenv("SFT_OUTPUT", "./sft_output"))
SFT_OUTPUT_PEFT = _resolve_path(os.getenv("SFT_OUTPUT_PEFT", "./sft_output_peft"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

HF_TOKEN = os.getenv("HF_TOKEN") or None
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
EVAL_MAX_SAMPLES = int(os.getenv("EVAL_MAX_SAMPLES", "256"))
NUM_TRAIN_EPOCHS = float(os.getenv("NUM_TRAIN_EPOCHS", "2"))
# 0: батчи собирает основной процесс. На CUDA безопаснее, чем fork после .to("cuda").
DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "0"))
# Если Good выиграл, но от Neutral меньше этого зазора — берём Neutral.
NEUTRAL_MARGIN = float(os.getenv("NEUTRAL_MARGIN", "0.15"))
