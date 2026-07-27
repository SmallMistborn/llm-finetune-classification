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
