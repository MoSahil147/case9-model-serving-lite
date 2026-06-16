"""
Fine-tune the base sentiment model on new training data.

Called by the retrain GitHub Actions workflow when a PR modifies data/train.csv.
The fine-tuned model is saved to the output directory and then evaluated by
scripts/evaluate.py before any promotion decision is made.

Expected CSV format:
    text,label
    "This product is brilliant.",1
    "Complete waste of money.",0

Where label 1 = POSITIVE and label 0 = NEGATIVE.

Usage:
    python scripts/retrain.py --train data/train.csv --output model/

Runtime note:
    Fine-tuning DistilBERT for 2 epochs on ~200 rows takes about 8-12 minutes
    on a GitHub Actions free runner (2 vCPUs). This is well within the 6-hour
    job limit. Increasing num_train_epochs will increase accuracy but also
    increase CI run time proportionally.
"""

import argparse
import csv
import os
import sys

# Fail fast: if HuggingFace libraries aren't installed, exit with a clear message
# rather than crashing mid-training with a confusing traceback.
try:
    from datasets import Dataset  # HuggingFace Dataset wrapper around our CSV rows
    from transformers import (
        AutoModelForSequenceClassification,  # loads a pre-trained model for text classification
        AutoTokenizer,  # loads the matching tokenizer for the model
        Trainer,  # high-level training loop (handles batching, backprop, etc.)
        TrainingArguments,  # config object for hyperparameters and output settings
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: uv sync")
    sys.exit(1)


# Which pre-trained model to start from. Can be overridden by setting the
# MODEL_NAME env var (e.g. to switch to a larger BERT variant without code changes).
BASE_MODEL = os.getenv(
    "MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english",  # default: DistilBERT already fine-tuned on SST-2 sentiment
)


def load_csv(path: str) -> list[dict]:
    """Read the training CSV and return a list of {text, label} dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            f
        )  # parses header row automatically; each row is a dict
        for row in reader:
            text = row.get("text", "").strip()
            label_raw = row.get("label", "").strip()

            # Only accept rows where label is exactly "0" or "1".
            # Anything else (missing column, typo, empty string) is silently skipped
            # so one bad row doesn't abort the whole training run.
            if not text or label_raw not in ("0", "1"):
                print(f"Skipping malformed row: {row}")
                continue

            # Convert label string to int: "0" → 0 (NEGATIVE), "1" → 1 (POSITIVE)
            rows.append({"text": text, "label": int(label_raw)})

    return rows


def tokenise(batch: dict, tokenizer) -> dict:
    """Tokenise a batch of texts. Called by the HuggingFace Dataset.map() method.

    Tokenization converts raw strings into integer token IDs the model can process.
    - truncation=True:       clips texts longer than max_length (avoids memory errors)
    - padding="max_length":  pads shorter texts with [PAD] tokens so all tensors are the same shape
    - max_length=128:        128 tokens covers most short reviews; longer texts are cut off
    """
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=128
    )


def main():
    # --- 1. Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Fine-tune the sentiment model on new data."
    )
    parser.add_argument(
        "--train", required=True, help="Path to training CSV (text, label columns)."
    )
    parser.add_argument(
        "--output", required=True, help="Directory to save the fine-tuned model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
        # 2 epochs is a sweet spot: enough to adapt to new data, fast enough for CI runners.
    )
    args = parser.parse_args()

    # --- 2. Load and validate training data ---
    print(f"Loading training data from: {args.train}")
    rows = load_csv(args.train)

    # Guard: refuse to fine-tune on a trivially small dataset — the model would
    # overfit badly and the resulting weights would be meaningless.
    if len(rows) < 10:
        print(
            f"ERROR: only {len(rows)} valid rows found. Need at least 10 to fine-tune."
        )
        sys.exit(1)

    print(f"Loaded {len(rows)} training examples.")

    # --- 3. Wrap rows in a HuggingFace Dataset ---
    # Dataset.from_list() converts a plain Python list of dicts into a columnar
    # Dataset object that Trainer and .map() understand natively.
    dataset = Dataset.from_list(rows)

    # --- 4. Load the pre-trained base model and its tokenizer ---
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # num_labels=2 matches our binary task: POSITIVE (1) vs NEGATIVE (0).
    # The pre-trained classification head already has 2 outputs for SST-2, so weights are reused.
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    # --- 5. Tokenise the entire dataset upfront ---
    # Doing this before training (rather than on-the-fly per batch) means the GPU/CPU
    # is never idle waiting for tokenization; it can just pull pre-processed tensors.
    tokenised = dataset.map(
        lambda batch: tokenise(batch, tokenizer),
        batched=True,  # process multiple rows at once (faster than row-by-row)
        remove_columns=["text"],  # drop raw text column; the model only needs token IDs
    )
    tokenised.set_format("torch")  # return PyTorch tensors instead of Python lists

    # --- 6. Configure training hyperparameters ---
    training_args = TrainingArguments(
        output_dir=args.output,  # where to write checkpoints and final model
        num_train_epochs=args.epochs,  # how many full passes over the training data
        per_device_train_batch_size=16,  # 16 samples per gradient update step
        # No mid-training evaluation — the held-out test set is checked separately
        # by evaluate.py after training completes. This keeps training fast.
        evaluation_strategy="no",
        save_strategy="epoch",  # checkpoint after every epoch (allows recovery)
        load_best_model_at_end=False,  # irrelevant since we don't evaluate mid-training
        logging_steps=10,  # print training loss every 10 steps
        fp16=False,  # disable half-precision; CI runners have no CUDA GPU
        report_to="none",  # don't push metrics to W&B, TensorBoard, etc.
    )

    # --- 7. Build the Trainer and run fine-tuning ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised,  # only a train split; no eval_dataset needed here
    )

    print(f"Starting fine-tuning for {args.epochs} epoch(s)...")
    trainer.train()  # blocks until all epochs are complete

    # --- 8. Persist the fine-tuned model to disk ---
    # Saving model + tokenizer together means any downstream code (evaluate.py,
    # the Docker API image) can load both with a single from_pretrained(output_dir) call.
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
