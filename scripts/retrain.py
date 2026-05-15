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

# We import these here so the script fails fast with a clear error message
# if the dev dependencies are not installed, rather than mid-training.
try:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements-dev.txt")
    sys.exit(1)


BASE_MODEL = os.getenv(
    "MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english",
)


def load_csv(path: str) -> list[dict]:
    """Read the training CSV and return a list of {text, label} dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label_raw = row.get("label", "").strip()

            if not text or label_raw not in ("0", "1"):
                # Skip malformed rows rather than crashing the whole run.
                print(f"Skipping malformed row: {row}")
                continue

            rows.append({"text": text, "label": int(label_raw)})

    return rows


def tokenise(batch: dict, tokenizer) -> dict:
    """Tokenise a batch of texts. Called by the HuggingFace Dataset.map() method."""
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=128
    )


def main():
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
        "--epochs", type=int, default=2, help="Number of training epochs."
    )
    args = parser.parse_args()

    print(f"Loading training data from: {args.train}")
    rows = load_csv(args.train)

    if len(rows) < 10:
        print(
            f"ERROR: only {len(rows)} valid rows found. Need at least 10 to fine-tune."
        )
        sys.exit(1)

    print(f"Loaded {len(rows)} training examples.")

    # HuggingFace Dataset is more ergonomic with the Trainer than plain lists.
    dataset = Dataset.from_list(rows)

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    # Tokenise all examples upfront so training is not bottlenecked on
    # on-the-fly tokenisation during each batch.
    tokenised = dataset.map(
        lambda batch: tokenise(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenised.set_format("torch")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        # We evaluate on the training set itself here — the real evaluation
        # against the held-out set happens separately in evaluate.py.
        # This is just to give us a training loss curve in the Trainer output.
        evaluation_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        logging_steps=10,
        # No GPU — disable mixed precision so we do not need CUDA.
        fp16=False,
        report_to="none",  # do not send to W&B or TensorBoard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised,
    )

    print(f"Starting fine-tuning for {args.epochs} epoch(s)...")
    trainer.train()

    # Save the final model and tokeniser together so evaluate.py and the
    # Docker image can load them with a single from_pretrained() call.
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
