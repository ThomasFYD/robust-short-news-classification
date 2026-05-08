import json
import os
from collections import Counter

from datasets import load_dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def combine_text(example):
    # AG News usually has a single "text" field.
    # If title/description style fields appear, combine them safely.
    if "text" in example and example["text"] is not None:
        return example["text"]
    title = example.get("title", "") or ""
    desc = example.get("description", "") or ""
    return f"{title} {desc}".strip()


def evaluate_model(name, y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_f1": {
            label: report[label]["f1-score"] for label in LABEL_NAMES
        },
    }
    return result


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading AG News...")
    dataset = load_dataset("ag_news")

    train_data = dataset["train"]
    test_data = dataset["test"]

    X_train = [combine_text(x) for x in train_data]
    y_train = [x["label"] for x in train_data]

    X_test = [combine_text(x) for x in test_data]
    y_test = [x["label"] for x in test_data]

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print("Train label distribution:", Counter(y_train))

    results = []

    # 1. Majority baseline
    print("\nTraining majority baseline...")
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(X_train, y_train)
    majority_preds = majority_clf.predict(X_test)
    majority_result = evaluate_model("majority_baseline", y_test, majority_preds)
    results.append(majority_result)

    # 2. TF-IDF + Logistic Regression
    print("\nTraining TF-IDF + Logistic Regression...")
    tfidf_lr = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=None,
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
    tfidf_lr_preds = tfidf_lr.predict(X_test)
    tfidf_lr_result = evaluate_model("tfidf_logreg", y_test, tfidf_lr_preds)
    results.append(tfidf_lr_result)

    # Save results
    output_path = "results/baseline_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to:", output_path)
    print("\nSummary:")
    for r in results:
        print(f"- {r['model']}: accuracy={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}")
        for label, f1 in r["per_class_f1"].items():
            print(f"    {label}: {f1:.4f}")


if __name__ == "__main__":
    main()