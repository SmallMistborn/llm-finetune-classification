from pathlib import Path
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from config import BASE_MODEL, SFT_OUTPUT, SFT_OUTPUT_PEFT
from get_device import get_device
from prepare_datasets import load_splits


def run_peft_training(
    max_steps=50,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    output_dir=None,
):
    device = get_device()
    output_path = Path(output_dir or SFT_OUTPUT_PEFT)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train, val, _test = load_splits(tokenizer)

    peft_config = LoraConfig(
        r=6,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_path),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        packing=False,
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_path))

    return {
        "mode": "peft",
        "base_model": BASE_MODEL,
        "output_dir": str(output_path),
        "max_steps": max_steps,
        "device": device,
    }


def run_full_training(
    max_steps=50,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    output_dir=None,
):
    device = get_device()
    output_path = Path(output_dir or SFT_OUTPUT)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train, val, _test = load_splits(tokenizer)

    training_args = SFTConfig(
        output_dir=str(output_path),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        packing=False,
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_path))

    return {
        "mode": "full",
        "base_model": BASE_MODEL,
        "output_dir": str(output_path),
        "max_steps": max_steps,
        "device": device,
    }
