from functools import partial

from datasets import load_dataset

from formatting_func import format_example


def load_splits(tokenizer=None):
    """
    Загружает kinopoisk и делит: train 70% / val 15% / test 15%.
    Если передан tokenizer — train и val сразу в prompt/completion для SFT.
    """
    data = load_dataset("blinoff/kinopoisk", split="train")
    split1 = data.train_test_split(test_size=0.3, seed=42)
    train = split1["train"]
    split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
    test = split2["train"]
    val = split2["test"]

    if tokenizer is not None:
        fmt = partial(format_example, tokenizer=tokenizer)
        train = train.map(fmt, remove_columns=train.column_names)
        val = val.map(fmt, remove_columns=val.column_names)

    return train, val, test
