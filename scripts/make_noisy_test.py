import json
import os
import random
import re
import string

from datasets import load_dataset


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def get_text(example):
    if "text" in example and example["text"] is not None:
        return example["text"]
    title = example.get("title", "") or ""
    desc = example.get("description", "") or ""
    return f"{title} {desc}".strip()


def random_capitalization(text, rng, prob=0.12):
    words = text.split()
    new_words = []

    for w in words:
        if rng.random() < prob:
            mode = rng.choice(["upper", "lower", "title"])
            if mode == "upper":
                new_words.append(w.upper())
            elif mode == "lower":
                new_words.append(w.lower())
            else:
                new_words.append(w.title())
        else:
            new_words.append(w)

    return " ".join(new_words)


def remove_some_punctuation(text, rng, prob=0.35):
    kept = []
    for ch in text:
        if ch in string.punctuation and rng.random() < prob:
            continue
        kept.append(ch)
    return "".join(kept)


def add_extra_spaces(text, rng, prob=0.08):
    chars = []
    for ch in text:
        chars.append(ch)
        if ch == " " and rng.random() < prob:
            chars.append(" ")
        elif ch in [".", ",", ";", ":", "!", "?"] and rng.random() < prob:
            chars.append(" ")
    return "".join(chars)


def typo_in_word(word, rng):
    if len(word) < 4:
        return word

    op = rng.choice(["swap", "delete", "duplicate"])

    if op == "swap" and len(word) >= 4:
        i = rng.randint(1, len(word) - 2)
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    if op == "delete" and len(word) >= 5:
        i = rng.randint(1, len(word) - 2)
        return word[:i] + word[i + 1 :]

    if op == "duplicate":
        i = rng.randint(1, len(word) - 2)
        return word[:i] + word[i] + word[i:]

    return word


def add_synthetic_typos(text, rng, prob=0.05):
    parts = re.findall(r"\w+|\W+", text)
    new_parts = []

    for part in parts:
        if part.isalpha() and rng.random() < prob:
            new_parts.append(typo_in_word(part, rng))
        else:
            new_parts.append(part)

    return "".join(new_parts)


def make_noisy_text(text, rng):
    noisy = text
    noisy = random_capitalization(noisy, rng, prob=0.12)
    noisy = remove_some_punctuation(noisy, rng, prob=0.35)
    noisy = add_extra_spaces(noisy, rng, prob=0.08)
    noisy = add_synthetic_typos(noisy, rng, prob=0.05)
    noisy = re.sub(r"[ \t]+", " ", noisy).strip()
    return noisy


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(42)

    os.makedirs("data", exist_ok=True)

    print("Loading AG News test split...")
    dataset = load_dataset("ag_news")
    test_data = dataset["test"]

    clean_records = []
    noisy_records = []

    for idx, ex in enumerate(test_data):
        text = get_text(ex)
        label_id = int(ex["label"])
        label_name = LABEL_NAMES[label_id]
        noisy_text = make_noisy_text(text, rng)

        clean_records.append(
            {
                "id": idx,
                "label": label_id,
                "label_name": label_name,
                "text": text,
            }
        )

        noisy_records.append(
            {
                "id": idx,
                "label": label_id,
                "label_name": label_name,
                "original_text": text,
                "noisy_text": noisy_text,
            }
        )

    clean_path = "data/ag_news_test_clean.jsonl"
    noisy_path = "data/ag_news_test_noisy.jsonl"
    summary_path = "data/ag_news_noisy_summary.json"

    save_jsonl(clean_path, clean_records)
    save_jsonl(noisy_path, noisy_records)

    summary = {
        "num_examples": len(noisy_records),
        "clean_file": clean_path,
        "noisy_file": noisy_path,
        "noise_types": [
            "random capitalization changes",
            "partial punctuation removal",
            "extra spaces",
            "limited synthetic typos",
        ],
        "random_seed": 42,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved clean test set to: {clean_path}")
    print(f"Saved noisy test set to: {noisy_path}")
    print(f"Saved summary to: {summary_path}")

    print("\nSample example:")
    print("Original:", noisy_records[0]["original_text"])
    print("Noisy:   ", noisy_records[0]["noisy_text"])


if __name__ == "__main__":
    main()