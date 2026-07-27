from pathlib import Path

from config import SFT_OUTPUT
from get_device import get_device
from inference import load_model, predict
from prepare_datasets import load_splits
from training import run_full_training

if not Path(SFT_OUTPUT).exists():
    run_full_training()

device = get_device()
model, tokenizer, _ = load_model(SFT_OUTPUT, device)
_train, _val, test = load_splits()

for ex in test:
    pred = predict(model, tokenizer, ex["movie_name"], ex["content"], device)
    print(ex["grade3"], pred, ex["movie_name"])
