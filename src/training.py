from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from config import (
    BASE_MODEL,
    DATALOADER_WORKERS,
    EVAL_MAX_SAMPLES,
    HF_TOKEN,
    MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS,
    SFT_OUTPUT,
    SFT_OUTPUT_PEFT,
)
from get_device import get_device
from prepare_datasets import load_splits


def _hub_kwargs() -> dict:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def _sft_config(
    output_path: Path,
    device: str,
    batch_size: int,
    learning_rate: float,
    max_steps: int | None,
    num_train_epochs: float,
):
    use_cuda = device == "cuda"
    args = {
        "output_dir": str(output_path),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "logging_steps": 20,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "packing": False,
        "max_length": MAX_SEQ_LENGTH,
        "completion_only_loss": True,
        "shuffle_dataset": True,
        "bf16": use_cuda,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "dataloader_num_workers": DATALOADER_WORKERS,
        "dataloader_pin_memory": use_cuda,
        "num_train_epochs": num_train_epochs,
    }
    if max_steps and max_steps > 0:
        args["max_steps"] = max_steps
        args["eval_strategy"] = "steps"
        args["save_strategy"] = "steps"
        args["eval_steps"] = max_steps
        args["save_steps"] = max_steps
        args["warmup_steps"] = max(1, int(0.05 * max_steps))
    else:
        args["max_steps"] = -1
        args["eval_strategy"] = "epoch"
        args["save_strategy"] = "epoch"
        args["warmup_steps"] = 50
    return SFTConfig(**args)


def _load_base(device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        attn_implementation="sdpa",
        **_hub_kwargs(),
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **_hub_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    return model, tokenizer


def _slice_eval(val):
    if EVAL_MAX_SAMPLES and len(val) > EVAL_MAX_SAMPLES:
        return val.shuffle(seed=42).select(range(EVAL_MAX_SAMPLES))
    return val


def _run_sft(
    mode: str,
    output_path: Path,
    batch_size: int,
    learning_rate: float,
    max_steps: int | None,
    num_train_epochs: float,
    peft_config=None,
):
    device = get_device()
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            f"CUDA: {torch.cuda.get_device_name(0)} "
            f"{props.total_memory / 1024 ** 3:.1f} GB, bf16={torch.cuda.is_bf16_supported()}"
        )
    else:
        print(
            f"WARNING: CUDA недоступна, обучение идёт на {device}. "
            "На Immers сначала: pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu124"
        )
    model, tokenizer = _load_base(device)
    param_device = next(model.parameters()).device
    print(f"model device={param_device}, dtype={next(model.parameters()).dtype}, base={BASE_MODEL}")
    if device == "cuda" and param_device.type != "cuda":
        raise RuntimeError(f"Модель не на GPU: {param_device}")
    train, val, _test = load_splits(tokenizer)
    val = _slice_eval(val)
    trainer_kwargs = {
        "model": model,
        "args": _sft_config(
            output_path, device, batch_size, learning_rate, max_steps, num_train_epochs
        ),
        "train_dataset": train,
        "eval_dataset": val,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_path))
    return {
        "mode": mode,
        "base_model": BASE_MODEL,
        "output_dir": str(output_path),
        "max_steps": max_steps,
        "num_train_epochs": num_train_epochs,
        "train_size": len(train),
        "eval_size": len(val),
        "max_length": MAX_SEQ_LENGTH,
        "device": device,
    }


def run_peft_training(
    max_steps=None,
    num_train_epochs=None,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    output_dir=None,
):
    output_path = Path(output_dir or SFT_OUTPUT_PEFT)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return _run_sft(
        mode="peft",
        output_path=output_path,
        batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs or NUM_TRAIN_EPOCHS,
        peft_config=peft_config,
    )


def run_full_training(
    max_steps=None,
    num_train_epochs=None,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    output_dir=None,
):
    output_path = Path(output_dir or SFT_OUTPUT)
    return _run_sft(
        mode="full",
        output_path=output_path,
        batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs or NUM_TRAIN_EPOCHS,
    )
