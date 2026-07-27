from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL
from formatting_func import build_prompt


def load_model(model_path, device=None):
    path = Path(model_path)

    if (path / "adapter_config.json").exists():
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(path),
            torch_dtype=torch.float16,
        )
        model = peft_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model_type = "peft"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model_type = "full"

    model.eval()
    return model, tokenizer, model_type


def predict(model, tokenizer, movie_name, review, device=None, max_new_tokens=8):
    prompt = build_prompt(tokenizer, movie_name, review)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()
