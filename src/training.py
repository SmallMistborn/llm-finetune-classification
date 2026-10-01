from pathlib import Path

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from config import BASE_MODEL, SFT_OUTPUT, SFT_OUTPUT_PEFT
from get_device import get_device
from prepare_datasets import load_splits


def _sft_config(output_path: Path, device: str, max_steps: int, batch_size: int, learning_rate: float):
    use_cuda = device == "cuda"
    eval_steps = max(1, min(100, max_steps))
    logging_steps = max(1, min(20, max_steps))
    warmup_steps = max(1, int(0.05 * max_steps))
    return SFTConfig(
        output_dir=str(output_path),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=eval_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
        max_length=1024,
        completion_only_loss=True,
        shuffle_dataset=True,
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=True,
    )


def _load_base(device: str):
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _run_sft(
    mode: str,
    output_path: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    peft_config=None,
):
    device = get_device()
    model, tokenizer = _load_base(device)
    train, val, _test = load_splits(tokenizer)
    trainer_kwargs = {
        "model": model,
        "args": _sft_config(output_path, device, max_steps, batch_size, learning_rate),
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
        "train_size": len(train),
        "device": device,
    }


def run_peft_training(
    max_steps=1000,
    per_device_train_batch_size=4,
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
        max_steps=max_steps,
        batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        peft_config=peft_config,
    )


def run_full_training(
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    output_dir=None,
):
    output_path = Path(output_dir or SFT_OUTPUT)
    return _run_sft(
        mode="full",
        output_path=output_path,
        max_steps=max_steps,
        batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
    )
