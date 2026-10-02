from threading import Lock

import torch
from sklearn.metrics import classification_report, f1_score

from config import NEUTRAL_MARGIN
from formatting_func import LABELS
from get_device import get_device
from inference import load_model, predict
from model_path import resolve_model_path
from prepare_datasets import load_splits


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
        path = str(resolve_model_path(model_path))
        with self.lock:
            if self.model is not None and self.model_path == path:
                return
            self.model, self.tokenizer, self.model_type = load_model(path)
            self.model_path = path

    def reload(self, model_path=None):
        self.unload()
        self.load(model_path)
        return self.status()

    def predict(self, movie_name, review, model_path=None):
        self.load(model_path)
        label, scores = predict(self.model, self.tokenizer, movie_name, review)
        return {
            "prediction": label,
            "scores": scores,
            "neutral_margin": NEUTRAL_MARGIN,
            "model_path": self.model_path,
            "model_type": self.model_type,
        }

    def evaluate(self, model_path=None, limit=20):
        self.load(model_path)
        _train, _val, test = load_splits()
        if limit:
            test = test.select(range(min(limit, len(test))))

        labels = list(LABELS)
        y_true, y_pred = [], []
        for example in test:
            label, _scores = predict(
                self.model,
                self.tokenizer,
                example["movie_name"],
                example["content"],
            )
            y_true.append(example["grade3"])
            y_pred.append(label)

        n = len(y_true)
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        return {
            "model": self.model_path,
            "model_type": self.model_type,
            "neutral_margin": NEUTRAL_MARGIN,
            "split": "test",
            "accuracy": correct / n if n else 0.0,
            "f1_macro": float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "f1_per_class": {
                label: float(score) for label, score in zip(labels, per_class)
            },
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, zero_division=0, output_dict=True
            ),
            "total": n,
            "correct": correct,
        }


model_service = ModelService()
