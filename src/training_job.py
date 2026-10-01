from threading import Lock, Thread

from model_path import default_output_dir
from model_service import model_service
from training import run_full_training, run_peft_training


class TrainingAlreadyRunningError(Exception):
    def __init__(self, message: str = "Идёт обучение. Дождитесь завершения или проверьте /train/status"):
        super().__init__(message)


class TrainingService:
    def __init__(self):
        self._lock = Lock()
        self._status = {
            "status": "idle",
            "mode": None,
            "output_dir": None,
            "error": None,
            "result": None,
        }

    def status(self) -> dict:
        return dict(self._status)

    def is_running(self) -> bool:
        return self._status["status"] == "running"

    def require_idle(self) -> None:
        if self.is_running():
            raise TrainingAlreadyRunningError()

    def start(
        self,
        mode: str,
        max_steps: int,
        per_device_train_batch_size: int,
        learning_rate: float,
        output_dir: str | None = None,
    ) -> dict:
        output_path = output_dir or default_output_dir(mode)
        params = {
            "max_steps": max_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "learning_rate": learning_rate,
            "output_dir": output_path,
        }
        with self._lock:
            if self.is_running():
                raise TrainingAlreadyRunningError("Обучение уже запущено")
            self._status = {
                "status": "running",
                "mode": mode,
                "output_dir": output_path,
                "error": None,
                "result": None,
            }

        model_service.unload()
        Thread(target=self._run, args=(mode, params), daemon=True).start()
        return {
            "message": "Обучение запущено",
            "mode": mode,
            "output_dir": output_path,
            "training": self.status(),
        }

    def _run(self, mode: str, params: dict) -> None:
        try:
            if mode == "peft":
                result = run_peft_training(**params)
            elif mode == "full":
                result = run_full_training(**params)
            else:
                raise ValueError(f"Неизвестный mode: {mode}")
            self._status = {
                "status": "completed",
                "mode": mode,
                "output_dir": result["output_dir"],
                "error": None,
                "result": result,
            }
            model_service.unload()
        except Exception as exc:
            self._status = {
                "status": "failed",
                "mode": mode,
                "output_dir": params.get("output_dir"),
                "error": str(exc),
                "result": None,
            }


training_service = TrainingService()
