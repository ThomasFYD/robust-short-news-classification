import json
import os

import torch
from datasets import load_dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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


def evaluate_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_f1": {
            label: report[label]["f1-score"] for label in LABEL_NAMES
        },
    }


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


def evaluate_distilbert(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            total_steps += 1
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    metrics = evaluate_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(total_steps, 1)
    metrics["world_prediction_rate"] = sum(1 for p in all_preds if p == 0) / max(len(all_preds), 1)
    return metrics


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading AG News train split...")
    dataset = load_dataset("ag_news")
    train_data = dataset["train"]

    X_train = [combine_text(x) for x in train_data]
    y_train = [x["label"] for x in train_data]

    print("Loading adversarial file to recover original clean subset...")
    adv_records = read_jsonl("data/ag_news_test_adversarial_worldcue.jsonl")

    X_clean_subset = [x["original_text"] for x in adv_records]
    y_clean_subset = [x["label"] for x in adv_records]

    results = {}

    print("\nEvaluating majority baseline...")
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(X_train, y_train)
    majority_preds = majority_clf.predict(X_clean_subset)
    results["majority_baseline"] = evaluate_metrics(y_clean_subset, majority_preds)

    print("Evaluating TF-IDF + Logistic Regression...")
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
    tfidf_preds = tfidf_lr.predict(X_clean_subset)
    results["tfidf_logreg"] = evaluate_metrics(y_clean_subset, tfidf_preds)

    print("Evaluating DistilBERT...")
    model_dir = "models/distilbert_ag_news"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    clean_dataset = TextDataset(X_clean_subset, y_clean_subset, tokenizer)
    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results["distilbert"] = evaluate_distilbert(model, clean_loader, device)

    output_path = "results/clean_subset_for_adversarial.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")
    print("\nSummary:")
    for model_name, metrics in results.items():
        if "loss" in metrics:
            print(
                f"{model_name} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f} | "
                f"world_prediction_rate={metrics['world_prediction_rate']:.4f}"
            )
        else:
            print(
                f"{model_name} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )


if __name__ == "__main__":
    main()