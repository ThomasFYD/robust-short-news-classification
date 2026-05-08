import json
import os
import random

from datasets import load_dataset


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

WORLD_CUE_SENTENCES = [
    "Officials in France commented on the situation.",
    "Government sources in Germany are monitoring developments.",
    "International observers in Japan issued a brief statement.",
    "Authorities in Canada said they were following the matter.",
    "Diplomats in the United Kingdom discussed the issue.",
    "Foreign ministry officials in Italy released a short update.",
    "Observers in Brazil said the development was being watched closely.",
    "Officials in Australia noted the issue in a public statement.",
]


def get_text(example):
    if "text" in example and example["text"] is not None:
        return example["text"]
    title = example.get("title", "") or ""
    desc = example.get("description", "") or ""
    return f"{title} {desc}".strip()


def add_world_cue(text, rng):
    cue = rng.choice(WORLD_CUE_SENTENCES)
    text = text.strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return text + " " + cue


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

    adv_records = []
    kept_labels = {"Sports", "Business", "Sci/Tech"}

    for idx, ex in enumerate(test_data):
        label_id = int(ex["label"])
        label_name = LABEL_NAMES[label_id]
        text = get_text(ex)

        if label_name not in kept_labels:
            continue

        adversarial_text = add_world_cue(text, rng)

        adv_records.append(
            {
                "id": idx,
                "label": label_id,
                "label_name": label_name,
                "original_text": text,
                "adversarial_text": adversarial_text,
                "attack_type": "world_cue_injection",
            }
        )

    output_path = "data/ag_news_test_adversarial_worldcue.jsonl"
    summary_path = "data/ag_news_adversarial_summary.json"

    save_jsonl(output_path, adv_records)

    summary = {
        "num_examples": len(adv_records),
        "output_file": output_path,
        "attack_type": "world_cue_injection",
        "source_labels": ["Sports", "Business", "Sci/Tech"],
        "target_confusion_hypothesis": "Examples may be misclassified as World after injection of spurious international/government cues.",
        "random_seed": 42,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved adversarial set to: {output_path}")
    print(f"Saved summary to: {summary_path}")

    print("\nSample example:")
    print("Original:    ", adv_records[0]["original_text"])
    print("Adversarial: ", adv_records[0]["adversarial_text"])


if __name__ == "__main__":
    main()