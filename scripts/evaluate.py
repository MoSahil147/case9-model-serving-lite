"""
Evaluate a trained model against the held-out evaluation set.

This script is always run against data/eval.csv which never changes on a
training data PR. That means the gate is comparing apples to apples: same
eval set, different model weights.

Outputs a JSON file at --out with accuracy and F1 score. The check_metrics.py
script reads this file and decides whether to promote the new model.

Usage:
    python scripts/evaluate.py \
        --model model/ \
        --eval  data/eval.csv \
        --out   metrics/candidate.json

Expected CSV format: same as train.csv (text, label columns).
"""

import argparse
import csv
import json
import sys

try:
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import pipeline
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements-dev.txt")
    sys.exit(1)


def load_csv(path: str) -> tuple[list[str], list[int]]:
    """Return (texts, labels) loaded from the evaluation CSV."""
    texts = []
    labels = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label_raw = row.get("label", "").strip()

            if not text or label_raw not in ("0", "1"):
                print(f"Skipping malformed row: {row}")
                continue

            texts.append(text)
            labels.append(int(label_raw))

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the held-out eval set.")
    parser.add_argument("--model", required=True, help="Path to the model directory (output of retrain.py).")
    parser.add_argument("--eval", required=True, help="Path to the evaluation CSV.")
    parser.add_argument("--out", required=True, help="Where to write the metrics JSON file.")
    args = parser.parse_args()

    print(f"Loading evaluation data from: {args.eval}")
    texts, true_labels = load_csv(args.eval)

    if len(texts) == 0:
        print("ERROR: evaluation file is empty or malformed.")
        sys.exit(1)

    print(f"Loaded {len(texts)} evaluation examples.")
    print(f"Loading model from: {args.model}")

    # Load the fine-tuned model from the directory saved by retrain.py.
    # device=-1 forces CPU — consistent with how the production API runs.
    classifier = pipeline(
        "sentiment-analysis",
        model=args.model,
        tokenizer=args.model,
        device=-1,
    )

    print("Running inference on evaluation set...")
    predictions_raw = classifier(texts, truncation=True, max_length=128, batch_size=32)

    # Convert the pipeline's label strings back to 0/1 integers
    # so we can use sklearn's metric functions.
    label_to_int = {"POSITIVE": 1, "NEGATIVE": 0}
    predicted_labels = [label_to_int[p["label"]] for p in predictions_raw]

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="macro")

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "n_eval_examples": len(texts),
    }

    print(f"Results: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to: {args.out}")


if __name__ == "__main__":
    main()
