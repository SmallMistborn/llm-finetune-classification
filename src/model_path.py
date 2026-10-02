from pathlib import Path

from config import PROJECT_ROOT, SFT_OUTPUT, SFT_OUTPUT_PEFT


class ModelNotFoundError(Exception):
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Модель не найдена: {path}")


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def resolve_model_path(model_path: str | None = None, default: Path | None = None) -> Path:
    path = resolve_path(model_path or default or SFT_OUTPUT_PEFT)
    if not path.exists():
        raise ModelNotFoundError(str(path))
    return path


def default_output_dir(mode: str) -> str:
    if mode == "peft":
        return str(SFT_OUTPUT_PEFT)
    return str(SFT_OUTPUT)
