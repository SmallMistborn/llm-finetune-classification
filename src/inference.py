from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL, HF_TOKEN
from formatting_func import LABELS, build_prompt


def _hub_kwargs() -> dict:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def _infer_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_model(model_path: str):
    path = Path(model_path)
    dtype = _infer_dtype()
    hub = _hub_kwargs()

    if (path / "adapter_config.json").exists():
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=dtype,
            device_map="auto",
            **hub,
        )
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(path),
        )
        model = peft_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(str(path), **hub)
        model_type = "peft"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            dtype=dtype,
            device_map="auto",
            **hub,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(path), **hub)
        model_type = "full"

    model.eval()
    return model, tokenizer, model_type


def _label_token_ids(tokenizer) -> dict[str, list[int]]:
    return {
        label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS
    }


def score_labels(model, tokenizer, prompt: str) -> dict[str, float]:
    """Length-normalized log-probability of each class string after the prompt."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = {k: v.to(model.device) for k, v in prompt_ids.items()}
    prompt_len = prompt_ids["input_ids"].shape[1]
    scores = {}

    for label, label_ids in _label_token_ids(tokenizer).items():
        label_tensor = torch.tensor([label_ids], device=model.device)
        input_ids = torch.cat([prompt_ids["input_ids"], label_tensor], dim=1)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(
            logits[0, prompt_len - 1 : prompt_len - 1 + len(label_ids)], dim=-1
        )
        token_logp = log_probs[
            torch.arange(len(label_ids), device=model.device), label_tensor[0]
        ]
        scores[label] = float(token_logp.mean().item())
    return scores


def predict(model, tokenizer, movie_name: str, review: str) -> tuple[str, dict[str, float]]:
    prompt = build_prompt(tokenizer, movie_name, review)
    scores = score_labels(model, tokenizer, prompt)
    return max(scores, key=scores.get), scores
