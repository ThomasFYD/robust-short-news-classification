import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_text(example):
    if "text" in example and example["text"] is not None:
        return example["text"]
    title = example.get("title", "") or ""
    desc = example.get("description", "") or ""
    return f"{title} {desc}".strip()


def preprocess_function(batch, tokenizer, max_length=128):
    texts = [get_text({"text": t}) for t in batch["text"]]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encodings["labels"] = batch["label"]
    return encodings


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
    set_seed(42)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_name = "distilbert-base-uncased"
    save_dir = "models/distilbert_ag_news"

    print("Loading dataset...")
    dataset = load_dataset("ag_news")

    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = train_val["train"]
    val_data = train_val["test"]
    test_data = dataset["test"]

    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")

    print("\nLoading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label={i: label for i, label in enumerate(LABEL_NAMES)},
        label2id={label: i for i, label in enumerate(LABEL_NAMES)},
    )

    print("Tokenizing datasets...")
    train_tok = train_data.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=train_data.column_names,
    )
    val_tok = val_data.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=val_data.column_names,
    )
    test_tok = test_data.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=test_data.column_names,
    )

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    batch_size = 16
    num_epochs = 2
    learning_rate = 2e-5
    weight_decay = 0.01

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tok, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tok, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\nUsing device: {device}")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_training_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 500 == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_loader)} | Loss {avg_loss:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch + 1} finished")
        print(f"Train loss: {train_loss:.4f}")
        print(
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val macro_f1: {val_metrics['macro_f1']:.4f}"
        )

    print("\nEvaluating on clean test set...")
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"Test accuracy: {test_metrics['accuracy']:.4f} | "
        f"Test macro_f1: {test_metrics['macro_f1']:.4f}"
    )
    for label, f1 in test_metrics["per_class_f1"].items():
        print(f"{label}: {f1:.4f}")

    print("\nSaving model/tokenizer...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    result = {
        "model": "distilbert-base-uncased",
        "task": "ag_news_clean_test",
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_length": 128,
        "device": str(device),
        "test_metrics": test_metrics,
        "model_save_dir": save_dir,
    }

    output_path = "results/distilbert_clean_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()