from config import SFT_OUTPUT_PEFT
from model_service import model_service
from prepare_datasets import load_splits
from training import run_peft_training

if not SFT_OUTPUT_PEFT.exists():
    run_peft_training()

_train, _val, test = load_splits()
for example in test:
    result = model_service.predict(example["movie_name"], example["content"])
    print(example["grade3"], result["prediction"], example["movie_name"])
