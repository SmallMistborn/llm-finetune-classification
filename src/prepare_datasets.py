from functools import partial

from datasets import ClassLabel, Value, concatenate_datasets, load_dataset

from formatting_func import LABELS, format_example


def _decode_grade3(dataset):
    # ClassLabel.map() re-encodes string labels back to ids; cast to int first.
    names = list(dataset.features["grade3"].names)
    dataset = dataset.cast_column("grade3", Value("int64"))
    return dataset.map(lambda example: {"grade3": names[int(example["grade3"])]})


def _balance_train(dataset, seed: int):
    """Downsample each class to the size of the rarest class."""
    parts = [
        dataset.filter(lambda example, value=label: example["grade3"] == value)
        for label in LABELS
    ]
    n_min = min(len(part) for part in parts)
    balanced = [part.shuffle(seed=seed).select(range(n_min)) for part in parts]
    return concatenate_datasets(balanced).shuffle(seed=seed)


def load_splits(tokenizer=None, balance_train: bool = True, seed: int = 42):
    """
    Загружает kinopoisk и делит: train 70% / val 15% / test 15%.
    Сплит стратифицирован по grade3.
    Train по умолчанию балансируется downsampling'ом до редкого класса.
    Если передан tokenizer — train и val сразу в prompt/completion для SFT.
    """
    data = load_dataset("blinoff/kinopoisk", split="train")
    data = data.cast_column("grade3", ClassLabel(names=list(LABELS)))

    split1 = data.train_test_split(
        test_size=0.3, seed=seed, stratify_by_column="grade3"
    )
    train = split1["train"]
    split2 = split1["test"].train_test_split(
        test_size=0.5, seed=seed, stratify_by_column="grade3"
    )
    test = split2["train"]
    val = split2["test"]

    train = _decode_grade3(train)
    val = _decode_grade3(val)
    test = _decode_grade3(test)

    if balance_train:
        train = _balance_train(train, seed=seed)

    if tokenizer is not None:
        fmt = partial(format_example, tokenizer=tokenizer)
        train = train.map(fmt, remove_columns=train.column_names)
        val = val.map(fmt, remove_columns=val.column_names)

    return train, val, test
