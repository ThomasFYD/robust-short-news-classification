import json
import os

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        labels=[0, 1, 2, 3],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_f1": {
            label: report[label]["f1-score"] for label in LABEL_NAMES
        },
        "confusion_matrix": cm.tolist(),
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

    world_pred_count = sum(1 for p in all_preds if p == 0)
    metrics["world_prediction_rate"] = world_pred_count / max(len(all_preds), 1)

    return metrics


def main():
    os.makedirs("results", exist_ok=True)

    model_dir = "models/distilbert_ag_news"
    print(f"Loading model/tokenizer from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    adv_records = read_jsonl("data/ag_news_test_adversarial_worldcue.jsonl")
    adv_texts = [x["adversarial_text"] for x in adv_records]
    adv_labels = [x["label"] for x in adv_records]

    adv_dataset = TextDataset(adv_texts, adv_labels, tokenizer)
    adv_loader = DataLoader(adv_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Using device: {device}")

    adv_metrics = evaluate(model, adv_loader, device)

    results = {
        "model": "distilbert-base-uncased",
        "setting": "adversarial_worldcue",
        "metrics": adv_metrics,
    }

    output_path = "results/distilbert_adversarial_worldcue.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAdversarial:")
    print(
        f"loss={adv_metrics['loss']:.4f} | "
        f"accuracy={adv_metrics['accuracy']:.4f} | "
        f"macro_f1={adv_metrics['macro_f1']:.4f} | "
        f"world_prediction_rate={adv_metrics['world_prediction_rate']:.4f}"
    )

    print("\nPer-class F1:")
    for label, f1 in adv_metrics["per_class_f1"].items():
        print(f"{label}: {f1:.4f}")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()