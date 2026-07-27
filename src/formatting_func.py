MAX_REVIEW_CHARS = 500


def build_user_content(movie_name: str, review: str) -> str:
    review = review[:MAX_REVIEW_CHARS]
    return (
        "Классифицируй отзыв о фильме по тональности.\n"
        "Ответь ровно одним словом: Bad, Good или Neutral.\n\n"
        f"Фильм: {movie_name}\n"
        f"Отзыв: {review}"
    )


def build_prompt(tokenizer, movie_name: str, review: str) -> str:
    messages = [{"role": "user", "content": build_user_content(movie_name, review)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_example(example, tokenizer):
    return {
        "prompt": build_prompt(tokenizer, example["movie_name"], example["content"]),
        "completion": example["grade3"],
    }