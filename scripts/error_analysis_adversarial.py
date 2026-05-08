import json
import os

import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def combine_text(example):
    if "text" in example and example["text"] is not None:
        return example["text"]
    title = example.get("title", "") or ""
    desc = example.get("description", "") or ""
    return f"{title} {desc}".strip()


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def predict_distilbert(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            preds.extend(batch_preds.cpu().tolist())
    return preds


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading training data for TF-IDF + LR...")
    dataset = load_dataset("ag_news")
    train_data = dataset["train"]

    X_train = [combine_text(x) for x in train_data]
    y_train = [x["label"] for x in train_data]

    print("Loading adversarial set...")
    adv_records = read_jsonl("data/ag_news_test_adversarial_worldcue.jsonl")

    clean_texts = [x["original_text"] for x in adv_records]
    adv_texts = [x["adversarial_text"] for x in adv_records]
    labels = [x["label"] for x in adv_records]

    # TF-IDF + LR
    print("Running TF-IDF + Logistic Regression...")
    tfidf_lr = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            max_features=50000,
            ngram_range=(1, 2),
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    tfidf_lr.fit(X_train, y_train)
    tfidf_clean_preds = tfidf_lr.predict(clean_texts).tolist()
    tfidf_adv_preds = tfidf_lr.predict(adv_texts).tolist()

    # DistilBERT
    print("Running DistilBERT...")
    model_dir = "models/distilbert_ag_news"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    clean_dataset = TextDataset(clean_texts, labels, tokenizer)
    adv_dataset = TextDataset(adv_texts, labels, tokenizer)

    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False)
    adv_loader = DataLoader(adv_dataset, batch_size=32, shuffle=False)

    distilbert_clean_preds = predict_distilbert(model, clean_loader, device)
    distilbert_adv_preds = predict_distilbert(model, adv_loader, device)

    analysis = {
        "tfidf_clean_correct_adv_wrong": [],
        "distilbert_clean_correct_adv_wrong": [],
    }

    for i, record in enumerate(adv_records):
        gold = labels[i]

        # TF-IDF cases
        if tfidf_clean_preds[i] == gold and tfidf_adv_preds[i] != gold:
            analysis["tfidf_clean_correct_adv_wrong"].append({
                "id": record["id"],
                "gold_label": LABEL_NAMES[gold],
                "clean_pred": LABEL_NAMES[tfidf_clean_preds[i]],
                "adversarial_pred": LABEL_NAMES[tfidf_adv_preds[i]],
                "original_text": record["original_text"],
                "adversarial_text": record["adversarial_text"],
            })

        # DistilBERT cases
        if distilbert_clean_preds[i] == gold and distilbert_adv_preds[i] != gold:
            analysis["distilbert_clean_correct_adv_wrong"].append({
                "id": record["id"],
                "gold_label": LABEL_NAMES[gold],
                "clean_pred": LABEL_NAMES[distilbert_clean_preds[i]],
                "adversarial_pred": LABEL_NAMES[distilbert_adv_preds[i]],
                "original_text": record["original_text"],
                "adversarial_text": record["adversarial_text"],
            })

    output_path = "results/error_analysis_adversarial.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\nSaved analysis to: {output_path}")
    print("\nCounts:")
    print("TF-IDF clean-correct -> adversarial-wrong:", len(analysis["tfidf_clean_correct_adv_wrong"]))
    print("DistilBERT clean-correct -> adversarial-wrong:", len(analysis["distilbert_clean_correct_adv_wrong"]))

    print("\nSample TF-IDF cases:")
    for ex in analysis["tfidf_clean_correct_adv_wrong"][:3]:
        print(f"- id={ex['id']} | gold={ex['gold_label']} | adv_pred={ex['adversarial_pred']}")

    print("\nSample DistilBERT cases:")
    for ex in analysis["distilbert_clean_correct_adv_wrong"][:3]:
        print(f"- id={ex['id']} | gold={ex['gold_label']} | adv_pred={ex['adversarial_pred']}")


if __name__ == "__main__":
    main()