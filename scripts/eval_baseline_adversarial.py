import json
import os

from datasets import load_dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


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


def evaluate_model(model_name, setting_name, y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": model_name,
        "setting": setting_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_f1": {
            label: report[label]["f1-score"] for label in LABEL_NAMES
        },
    }


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading AG News train split...")
    dataset = load_dataset("ag_news")
    train_data = dataset["train"]

    X_train = [combine_text(x) for x in train_data]
    y_train = [x["label"] for x in train_data]

    print("Loading adversarial test file...")
    adv_records = read_jsonl("data/ag_news_test_adversarial_worldcue.jsonl")

    X_adv = [x["adversarial_text"] for x in adv_records]
    y_adv = [x["label"] for x in adv_records]

    results = []

    print("\nTraining majority baseline...")
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(X_train, y_train)
    majority_preds = majority_clf.predict(X_adv)
    results.append(evaluate_model("majority_baseline", "adversarial_worldcue", y_adv, majority_preds))

    print("Training TF-IDF + Logistic Regression...")
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
    tfidf_preds = tfidf_lr.predict(X_adv)
    results.append(evaluate_model("tfidf_logreg", "adversarial_worldcue", y_adv, tfidf_preds))

    output_path = "results/baseline_adversarial_worldcue.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")
    print("\nSummary:")
    for r in results:
        print(f"{r['model']} | accuracy={r['accuracy']:.4f} | macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()