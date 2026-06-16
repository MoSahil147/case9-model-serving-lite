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

import argparse   # parses --model, --eval, --out from the command line
import csv        # reads the evaluation CSV row by row
import json       # writes the final metrics as a JSON file
import sys        # sys.exit(1) to abort with a non-zero error code

# Guarded imports — if these libraries aren't installed, exit immediately
# with a helpful message instead of crashing mid-evaluation.
try:
    from sklearn.metrics import accuracy_score, f1_score  # compute accuracy and F1
    from transformers import pipeline                      # high-level HuggingFace inference wrapper
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: uv sync")
    sys.exit(1)


def load_csv(path: str) -> tuple[list[str], list[int]]:
    """Return (texts, labels) loaded from the evaluation CSV.

    Unlike retrain.py's load_csv which returns a list of dicts,
    this returns two separate parallel lists — texts and labels —
    because sklearn's metric functions expect them as separate arguments.
    """
    texts = []   # raw text strings: ["Great product!", "Terrible.", ...]
    labels = []  # ground-truth integer labels: [1, 0, ...]

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # first row is treated as column headers
        for row in reader:
            text = row.get("text", "").strip()
            label_raw = row.get("label", "").strip()

            # Skip rows that are missing text or have an invalid label.
            # Only "0" (NEGATIVE) and "1" (POSITIVE) are valid.
            if not text or label_raw not in ("0", "1"):
                print(f"Skipping malformed row: {row}")
                continue

            texts.append(text)
            labels.append(int(label_raw))  # "1" → 1, "0" → 0

    return texts, labels  # two parallel lists: texts[i] has ground-truth label labels[i]


def main():
    # --- 1. Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the held-out eval set."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model directory (output of retrain.py).",
        # e.g. --model model/   (the folder retrain.py saved to)
    )
    parser.add_argument(
        "--eval",
        required=True,
        help="Path to the evaluation CSV.",
        # e.g. --eval data/eval.csv  (the held-out test set, never used in training)
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Where to write the metrics JSON file.",
        # e.g. --out metrics/candidate.json  (read later by check_metrics.py)
    )
    args = parser.parse_args()

    # --- 2. Load the evaluation CSV into parallel lists ---
    print(f"Loading evaluation data from: {args.eval}")
    texts, true_labels = load_csv(args.eval)
    # texts       = ["Great product!", "Waste of money.", ...]  (what the model will see)
    # true_labels = [1, 0, ...]                                 (the correct answers)

    # Guard: if the eval file is empty or all rows were malformed, abort.
    if len(texts) == 0:
        print("ERROR: evaluation file is empty or malformed.")
        sys.exit(1)

    print(f"Loaded {len(texts)} evaluation examples.")

    # --- 3. Load the fine-tuned model via HuggingFace pipeline ---
    print(f"Loading model from: {args.model}")
    classifier = pipeline(
        "sentiment-analysis",  # task type — tells pipeline to return {"label": "POSITIVE"/"NEGATIVE", "score": float}
        model=args.model,      # path to model/ directory saved by retrain.py
        tokenizer=args.model,  # same directory contains the tokenizer files (vocab.txt etc.)
        device_map="auto",     # auto-selects CPU/GPU/MPS — consistent with predict.py
    )
    # `classifier` is now a callable that accepts a list of strings and returns predictions

    # --- 4. Run inference on all evaluation texts ---
    print("Running inference on evaluation set...")
    predictions_raw = classifier(
        texts,              # the full list of evaluation strings
        truncation=True,    # clip texts longer than max_length (same setting used in training)
        max_length=128,     # must match the max_length used in retrain.py's tokenise()
        batch_size=32,      # process 32 texts at a time — faster than one-by-one
    )
    # predictions_raw looks like:
    # [
    #   {"label": "POSITIVE", "score": 0.998},
    #   {"label": "NEGATIVE", "score": 0.976},
    #   ...
    # ]

    # --- 5. Convert pipeline string labels back to integers ---
    # The pipeline returns "POSITIVE"/"NEGATIVE" strings, but sklearn's metric
    # functions need integers (1/0) to compare against true_labels.
    label_to_int = {"POSITIVE": 1, "NEGATIVE": 0}
    predicted_labels = [label_to_int[p["label"]] for p in predictions_raw]
    # predicted_labels = [1, 0, 1, 1, 0, ...]  — model's integer predictions

    # --- 6. Compute accuracy and F1 score ---
    accuracy = accuracy_score(true_labels, predicted_labels)
    # accuracy = correct_predictions / total_predictions
    # e.g. 92 correct out of 100 → accuracy = 0.92

    f1 = f1_score(true_labels, predicted_labels, average="macro")
    # F1 = harmonic mean of precision and recall, computed per class then averaged.
    # "macro" means: compute F1 for POSITIVE class + F1 for NEGATIVE class, then average them equally.
    # This penalises the model if it does well on one class but poorly on the other.

    # --- 7. Bundle metrics into a dict and round to 4 decimal places ---
    metrics = {
        "accuracy": round(accuracy, 4),          # e.g. 0.9250
        "f1": round(f1, 4),                      # e.g. 0.9198
        "n_eval_examples": len(texts),           # e.g. 200 — useful for audit/debugging
    }

    print(f"Results: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    # --- 8. Write metrics to a JSON file for check_metrics.py to consume ---
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Output file looks like:
    # {
    #   "accuracy": 0.925,
    #   "f1": 0.9198,
    #   "n_eval_examples": 200
    # }

    print(f"Metrics written to: {args.out}")


# Only call main() when this script is run directly.
# Prevents main() from running if another script does `import evaluate`.
if __name__ == "__main__":
    main()
