import json
import os

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
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


def evaluate(model, dataloader, device):
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

    metrics = build_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(total_steps, 1)
    return metrics


def main():
    os.makedirs("results", exist_ok=True)

    model_dir = "models/distilbert_ag_news"
    print(f"Loading model/tokenizer from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    clean_records = read_jsonl("data/ag_news_test_clean.jsonl")
    noisy_records = read_jsonl("data/ag_news_test_noisy.jsonl")

    clean_texts = [x["text"] for x in clean_records]
    clean_labels = [x["label"] for x in clean_records]

    noisy_texts = [x["noisy_text"] for x in noisy_records]
    noisy_labels = [x["label"] for x in noisy_records]

    clean_dataset = TextDataset(clean_texts, clean_labels, tokenizer)
    noisy_dataset = TextDataset(noisy_texts, noisy_labels, tokenizer)

    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False)
    noisy_loader = DataLoader(noisy_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Using device: {device}")

    clean_metrics = evaluate(model, clean_loader, device)
    noisy_metrics = evaluate(model, noisy_loader, device)

    results = {
        "model": "distilbert-base-uncased",
        "clean": clean_metrics,
        "noisy": noisy_metrics,
        "drops": {
            "accuracy_drop": clean_metrics["accuracy"] - noisy_metrics["accuracy"],
            "macro_f1_drop": clean_metrics["macro_f1"] - noisy_metrics["macro_f1"],
        },
    }

    output_path = "results/distilbert_clean_vs_noisy.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nClean:")
    print(f"loss={clean_metrics['loss']:.4f} | accuracy={clean_metrics['accuracy']:.4f} | macro_f1={clean_metrics['macro_f1']:.4f}")

    print("\nNoisy:")
    print(f"loss={noisy_metrics['loss']:.4f} | accuracy={noisy_metrics['accuracy']:.4f} | macro_f1={noisy_metrics['macro_f1']:.4f}")

    print("\nDrops:")
    print(f"accuracy_drop={results['drops']['accuracy_drop']:.4f}")
    print(f"macro_f1_drop={results['drops']['macro_f1_drop']:.4f}")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()