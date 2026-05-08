# Robust Short News Classification

This repository contains the code and report files for my CS505 NLP final project at Boston University.

## Project Overview

This project studies **robust short news classification** on the **AG News** dataset. The task is a 4-class topic classification problem with the labels:

- World
- Sports
- Business
- Sci/Tech

The main goal is to compare standard classification performance and robustness under two kinds of perturbations:

1. **Surface-level noise**, such as capitalization changes, punctuation removal, extra spaces, and synthetic typos
2. **Targeted adversarial cue injection**, where misleading World-style cue sentences are added to non-World examples

## Models

This project compares three levels of methods:

1. **Majority baseline**
2. **TF-IDF + Logistic Regression**
3. **DistilBERT** (`distilbert-base-uncased`)

## Experimental Settings

The models are evaluated on:

- **Clean test set**
- **Noisy test set**
- **Targeted adversarial subset**

The noisy test set is created by applying small realistic perturbations to the AG News test data.

The adversarial subset is created by injecting short World-style cue sentences into Sports, Business, and Sci/Tech examples in order to test whether the classifier is vulnerable to spurious correlations.

## Repository Structure

```text
.
├── data/
│   ├── ag_news_test_clean.jsonl
│   ├── ag_news_test_noisy.jsonl
│   ├── ag_news_test_adversarial_worldcue.jsonl
│   ├── ag_news_noisy_summary.json
│   └── ag_news_adversarial_summary.json
├── results/
│   ├── baseline_results.json
│   ├── baseline_clean_vs_noisy.json
│   ├── baseline_adversarial_worldcue.json
│   ├── distilbert_clean_results.json
│   ├── distilbert_clean_vs_noisy.json
│   ├── distilbert_adversarial_worldcue.json
│   ├── clean_subset_for_adversarial.json
│   └── error_analysis_adversarial.json
├── scripts/
│   ├── train_baseline.py
│   ├── train_distilbert.py
│   ├── make_noisy_test.py
│   ├── make_adversarial_test.py
│   ├── eval_baseline_noisy.py
│   ├── eval_distilbert_noisy.py
│   ├── eval_baseline_adversarial.py
│   ├── eval_distilbert_adversarial.py
│   ├── eval_clean_subset_for_adversarial.py
│   └── error_analysis_adversarial.py
├── FPP.pdf
├── MDWR.pdf
├── FFPR.pdf
└── README.md
