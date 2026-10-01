import argparse

from training import run_full_training, run_peft_training


def main():
    parser = argparse.ArgumentParser(description="SFT классификации отзывов Kinopoisk")
    parser.add_argument("--mode", choices=["peft", "full"], default="full")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    kwargs = {
        "max_steps": args.max_steps,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "output_dir": args.output_dir,
    }
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate

    runner = run_peft_training if args.mode == "peft" else run_full_training
    print(runner(**kwargs))


if __name__ == "__main__":
    main()
