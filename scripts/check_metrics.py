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

import argparse   # parses --baseline and --candidate paths from the command line
import json       # reads metric JSON files and pretty-prints them
import os         # reads GITHUB_OUTPUT env var and METRIC_TOLERANCE override
import sys        # sys.exit(0) for pass, sys.exit(1) for fail


# How much the new model is allowed to drop before we reject it.
# -0.02 = allow up to a 2 percentage point absolute drop in accuracy or F1.
# Can be overridden without code changes by setting METRIC_TOLERANCE env var.
# e.g. METRIC_TOLERANCE=-0.05 loosens the gate to allow a 5% drop.
REGRESSION_TOLERANCE = float(os.getenv("METRIC_TOLERANCE", "-0.02"))


def load_json(path: str) -> dict:
    """Read a JSON file from disk and return it as a Python dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # e.g. {"accuracy": 0.925, "f1": 0.919, "n_eval_examples": 200}


def set_github_output(key: str, value: str) -> None:
    # In a normal Python script, if you want to pass a value to the next step, you'd use a variable. 
    # But in GitHub Actions, each step runs in its own separate shell process — variables don't survive between steps. 
    # So GitHub invented a special mechanism: a shared temp file.
    """
    Write a key=value pair to the GITHUB_OUTPUT file so the workflow can
    read it with ${{ steps.<step-id>.outputs.<key> }}.

    Falls back to printing when running locally (GITHUB_OUTPUT is not set).
    """
    # GitHub Actions sets the GITHUB_OUTPUT env var to a temp file path.
    # Writing "promote=true\n" to that file makes it available as
    # ${{ steps.gate.outputs.promote }} in the next workflow steps.
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        # Running inside GitHub Actions — write to the special output file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    else:
        # Running locally — just print what would have been written
        print(f"[LOCAL] GITHUB_OUTPUT would be: {key}={value}")


def main():
    # --- 1. Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Decide whether to promote a retrained model."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the committed baseline metrics JSON.",
        # e.g. --baseline metrics/baseline.json
        # This is the current production model's scores — already in the repo.
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to the candidate metrics JSON from evaluate.py.",
        # e.g. --candidate metrics/candidate.json
        # This was just written by evaluate.py for the newly trained model.
    )
    args = parser.parse_args()

    # --- 2. Load both JSON files into dicts ---
    baseline  = load_json(args.baseline)   # current production model's scores
    candidate = load_json(args.candidate)  # newly trained model's scores

    # Print both side by side for visibility in the CI logs.
    # The dict comprehension {k: v for k, v in baseline.items() if k != "note"}
    # strips the optional human-readable "note" key from the baseline before printing.
    print("--- Baseline ---")
    print(json.dumps({k: v for k, v in baseline.items() if k != "note"}, indent=2))
    print("--- Candidate ---")
    print(json.dumps({k: v for k, v in candidate.items()}, indent=2))

    # --- 3. Check each metric for regression ---
    failures = []  # collect all failed metrics before deciding — don't stop at the first failure

    # Loop over both metrics we care about: accuracy and F1.
    for metric in ("accuracy", "f1"):
        baseline_value  = baseline.get(metric)   # e.g. 0.92
        candidate_value = candidate.get(metric)  # e.g. 0.89

        # If either file is missing this metric key, skip instead of crashing.
        # This guards against malformed JSON (e.g. evaluate.py was changed to
        # output different key names).
        if baseline_value is None or candidate_value is None:
            print(
                f"WARNING: '{metric}' is missing from one of the metric files — skipping check."
            )
            continue

        # delta = how much the new model changed vs the old model.
        # Positive delta → improvement. Negative delta → regression.
        # e.g. 0.89 - 0.92 = -0.03 (dropped 3 percentage points)
        delta = candidate_value - baseline_value

        print(
            f"\n{metric}: baseline={baseline_value:.4f}, candidate={candidate_value:.4f}, delta={delta:+.4f}"
            # :+.4f formats the delta with an explicit + or - sign, e.g. "+0.0100" or "-0.0300"
        )

        # Gate check: if delta is worse than the allowed tolerance, record a failure.
        # e.g. delta=-0.03  and  REGRESSION_TOLERANCE=-0.02
        #      -0.03 < -0.02  →  True  →  this metric failed
        if delta < REGRESSION_TOLERANCE:
            failures.append(
                f"{metric} dropped by {abs(delta):.4f} "
                f"(tolerance is {abs(REGRESSION_TOLERANCE):.2f})"
            )

    # --- 4. Make the final promote/reject decision ---
    if failures:
        # At least one metric regressed beyond tolerance → reject the new model.
        print("\nGATE FAILED — model regressed on the following metrics:")
        for reason in failures:
            print(f"  - {reason}")

        # Write promote=false to GitHub Actions output.
        # The workflow step `if: steps.gate.outputs.promote == 'false'`
        # uses this to trigger the reject/comment step.
        set_github_output("promote", "false")

        # Exit code 1 marks this CI step as failed — blocks the PR from merging.
        sys.exit(1)
    else:
        # All metrics are within tolerance → approve the new model.
        print("\nGATE PASSED — promoting new model.")

        # Write promote=true to GitHub Actions output.
        # The workflow step `if: steps.gate.outputs.promote == 'true'`
        # uses this to trigger the promote/merge step.
        set_github_output("promote", "true")

        # Exit code 0 marks this CI step as passed.
        sys.exit(0)


# Only call main() when this script is run directly.
# Prevents main() from auto-executing if another script imports this module.
if __name__ == "__main__":
    main()
