"""
Metric gate: compare a candidate model's metrics against the committed baseline.

This is the decision point in the retrain workflow. If the new model meets
the threshold we output promote=true to GITHUB_OUTPUT and exit 0. If it
regresses beyond the tolerance we output promote=false and exit 1, which
causes the GitHub Actions job to fail and blocks the PR from merging.

Thresholds:
    We allow up to a 2% absolute drop in accuracy OR F1.
    A 2% drop sounds small but on a binary classifier it can mean hundreds
    of misclassified users per day at production scale. Making this tighter
    is always safer.

Usage (from the GitHub Actions workflow):
    python scripts/check_metrics.py \
        --baseline metrics/baseline.json \
        --candidate metrics/candidate.json

Exit codes:
    0 — gate passed, promote=true written to GITHUB_OUTPUT
    1 — gate failed, promote=false written to GITHUB_OUTPUT
"""

import argparse
import json
import os
import sys


# Absolute tolerance for metric regression.
# -0.02 means we allow up to a 2 percentage point drop.
REGRESSION_TOLERANCE = float(os.getenv("METRIC_TOLERANCE", "-0.02"))


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_github_output(key: str, value: str) -> None:
    """
    Write a key=value pair to the GITHUB_OUTPUT file so the workflow can
    read it with ${{ steps.<step-id>.outputs.<key> }}.

    Falls back to printing when running locally (GITHUB_OUTPUT is not set).
    """
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    else:
        print(f"[LOCAL] GITHUB_OUTPUT would be: {key}={value}")


def main():
    parser = argparse.ArgumentParser(
        description="Decide whether to promote a retrained model."
    )
    parser.add_argument(
        "--baseline", required=True, help="Path to the committed baseline metrics JSON."
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to the candidate metrics JSON from evaluate.py.",
    )
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    print("--- Baseline ---")
    print(json.dumps({k: v for k, v in baseline.items() if k != "note"}, indent=2))
    print("--- Candidate ---")
    print(json.dumps({k: v for k, v in candidate.items()}, indent=2))

    failures = []

    for metric in ("accuracy", "f1"):
        baseline_value = baseline.get(metric)
        candidate_value = candidate.get(metric)

        if baseline_value is None or candidate_value is None:
            print(
                f"WARNING: '{metric}' is missing from one of the metric files — skipping check."
            )
            continue

        delta = candidate_value - baseline_value

        print(
            f"\n{metric}: baseline={baseline_value:.4f}, candidate={candidate_value:.4f}, delta={delta:+.4f}"
        )

        if delta < REGRESSION_TOLERANCE:
            failures.append(
                f"{metric} dropped by {abs(delta):.4f} "
                f"(tolerance is {abs(REGRESSION_TOLERANCE):.2f})"
            )

    if failures:
        print("\nGATE FAILED — model regressed on the following metrics:")
        for reason in failures:
            print(f"  - {reason}")
        set_github_output("promote", "false")
        sys.exit(1)
    else:
        print("\nGATE PASSED — promoting new model.")
        set_github_output("promote", "true")
        sys.exit(0)


if __name__ == "__main__":
    main()
