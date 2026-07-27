from pathlib import Path
from threading import Lock
import torch
from config import SFT_OUTPUT_PEFT
from get_device import get_device
from inference import load_model, predict


class ModelService:
    def __init__(self):
        self.lock = Lock()
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.model_path = None
        self.device = get_device()

    def status(self):
        if self.model is None:
            return {"loaded": False, "device": self.device}
        return {
            "loaded": True,
            "device": self.device,
            "model_path": self.model_path,
            "model_type": self.model_type,
        }

    def unload(self):
        with self.lock:
            self.model = None
            self.tokenizer = None
            self.model_type = None
            self.model_path = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, model_path=None):
        path = str(model_path or SFT_OUTPUT_PEFT)
        with self.lock:
            if self.model is not None and self.model_path == path:
                return
            self.model, self.tokenizer, self.model_type = load_model(path, self.device)
            self.model_path = path

    def predict(self, movie_name, review, model_path=None, max_new_tokens=8):
        self.load(model_path)
        text = predict(
            self.model,
            self.tokenizer,
            movie_name,
            review,
            self.device,
            max_new_tokens=max_new_tokens,
        )
        return {
            "prediction": text,
            "model_path": self.model_path,
            "model_type": self.model_type,
        }


model_service = ModelService()
