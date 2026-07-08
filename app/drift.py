# Drift means the real-world data coming into your model starts looking different from what it was trained on.

"""
Drift monitoring stub.

In production you would push these signals to Prometheus or InfluxDB or even Redis and
build dashboards around them. Here we do everything in memory so the demo
can show a live drift alert without any external infrastructure.

Three signals we track:

    mean_length     — average character length of recent requests.
                      A sudden drop (e.g., people sending single words instead
                      of sentences) means the model is being used differently.

    non_ascii_frac  — fraction of characters that are outside plain ASCII.
                      A spike usually means the input language has shifted.
                      DistilBERT was trained on English; non-English text will
                      produce unreliable sentiment scores.

    oov_rate        — fraction of whitespace-split tokens not seen in the
                      training corpus. A high OOV (Out Of Vocabulary) rate means the model is
                      encountering vocabulary it was not optimised for.

We compare a rolling window of recent requests against a baseline computed
from data/train.csv at startup. If any signal drifts more than DRIFT_THRESHOLD
(25% relative change by default) we mark the report as "alert".

Important caveats to mention in the writeup:
    - The window is per-process. With multiple workers each has its own window.
    - Restarting the container resets everything.
    - This is a stub; a real system would use Evidently or NannyML.
"""
# stub means works for demo purposes but too simple for real prod, Evi and Nanny are proffesional drift monitorying tools with proper stta, dashboards and alerting built in

import collections
import csv
import os
import re
import statistics
from pathlib import Path

# how many requests to look into
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW", "100"))

# 0.25 means a 25% relative shift from baseline triggers an alert.
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.25"))

# OOV rate above this absolute value is always flagged, regardless of baseline.
# if 30% words are unkown, Alert Alert
OOV_ALERT_THRESHOLD = 0.30
MIN_NON_ASCII_ALERT_DELTA = 0.01

# Production: DRIFT_WINDOW=500  → larger window
# Testing:    DRIFT_WINDOW=10   → smaller window
# No code changes needed


# we are creating a monitor to check Drift
class DriftMonitor:
    """
    Maintains a rolling window of input statistics and compares them against
    a baseline derived from the training corpus.
    """

    # runs automotically when you we create a DriftMonitor()
    def __init__(self, train_csv_path: str = "data/train.csv") -> None:
        # init sets up the monitor
        # record records each request
        # report checks the drift

        # deque automatically discards old entries once the window is full.
        # double ended queue, like a list but with max size, auto drops oldest entires when new ones are added
        self.window: collections.deque = collections.deque(maxlen=WINDOW_SIZE)

        # set because, its like list but no dups

        self._train_vocab: set = set()  # set of all words seen in training data, used to detect OOV words in live requests
        # set = no dups and very fast lookup

        # eveyrthing will be compared against the baseline and will store what a normal will look like
        self.baseline: dict = self._compute_baseline(
            Path(train_csv_path)
        )  # computes what "normal" looks like from training data!

    # what is nromal, reads the training CSV and compute normal stats
    def _compute_baseline(self, train_path: Path) -> dict:
        """
        Read the training CSV and compute the baseline distribution stats.

        If the file does not exist (e.g., during unit tests with no data folder)
        we fall back to sensible defaults so the app still starts cleanly.
        """
        #  safety check, no train.csv found, app will start cleanly and doesnt crash
        # dont crash, return some sensible things back
        if not train_path.exists():
            # Reasonable defaults for short English sentences.
            # if no training file found, will use sensible defaults instead of crashing
            self._train_vocab = set()
            # 80 chars length and 1% non ascii is acceptable
            return {
                "mean_length": 80.0,
                "non_ascii_frac": 0.01,
            }

        texts = []  # empty list to collect all the training texts
        # utf-8, represents texts as a sequenece of bytes
        with open(train_path, newline="", encoding="utf-8") as f:
            # 1st read ths CSV, {"text": "Amazing product!", "label": "POSITIVE"}
            reader = csv.DictReader(f)  # reads CSV where each row becomes a dict
            for row in reader:
                # 2nd just get the text column
                text = row.get(
                    "text", ""
                ).strip()  # will get text, deafults to empty string if column missing, strip is to remove the trailing spaces
                if text:  # skip empty rows
                    texts.append(text)
                    # Build vocabulary as lowercase whitespace-split tokens. first lower case, then split and add to vocab set
                    self._train_vocab.update(
                        re.findall(r"\b\w+\b", text.lower())
                    )  # findall scans the string and returns a list of all non overlapping matches
                    # "Amazing Product!" → "amazing product!" → ["amazing", "product!"]

        # if file exists but has no data, fall back to defaults
        if not texts:
            return {"mean_length": 80.0, "non_ascii_frac": 0.01}

        # mean length of each text or word
        mean_length = statistics.mean(len(t) for t in texts)
        # texts = ["Amazing!", "Terrible service!", "Good product"]
        # lengths = [8, 18, 12]
        # mean = (8 + 18 + 12) / 3 = 12.67

        # Count non-ASCII characters across all training texts.
        non_ascii_fracs = [
            # ord() is inbuilt Python fuction that takes a char and returns ist ASCII code
            sum(1 for c in t if ord(c) > 127) / max(len(t), 1)
            for t in texts  # "café"  → 'c','a','f','é'  → é is ord 233 > 127 → 1/4 = 0.25
        ]
        # "café"  → 'c','a','f','é'  → é is ord 233 > 127 → 1/4 = 0.25
        # "hello" → all ASCII → 0/5 = 0.0
        # mean_non_ascii = (0.25 + 0.0) / 2 = 0.125
        mean_non_ascii = statistics.mean(non_ascii_fracs)

        return {
            "mean_length": round(mean_length, 2),
            "non_ascii_frac": round(mean_non_ascii, 4),
        }

    # Log Each Request, called after every prediction
    def record(self, text: str) -> None:
        """
        Record statistics for one incoming request.
        Call this from the /predict endpoint after each successful inference.
        """
        words = re.findall(r"\b\w+\b", text.lower())

        # How many words in this request did we never see in training data?
        if words:
            # self._train_vocab, set of all words seen in training data
            #             Training vocab = {"amazing", "product", "terrible"}

            # Request: "fantastic cryptocurrency blockchain"
            # words   = ["fantastic", "cryptocurrency", "blockchain"]

            # "fantastic"    not in vocab → 1
            # "cryptocurrency" not in vocab → 1
            # "blockchain"   not in vocab → 1

            # oov_count = 3
            # oov_rate  = 3/3 = 1.0 (100% unknown!) ⚠️
            oov_count = sum(
                1 for w in words if w not in self._train_vocab
            )  # produces 1 for every word not in that set, then sums them up
            oov_rate = oov_count / len(words)
        else:
            oov_rate = 0.0

        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_frac = non_ascii_count / max(
            len(text), 1
        )  # max(len(text), 1) prevents division by zero on empty strings

        self.window.append(
            # window is a deque once it hits 100 entires the oldest one is automatically dropped
            {
                "length": len(text),
                "non_ascii_frac": non_ascii_frac,
                "oov_rate": oov_rate,
            }
        )

    def report(self) -> dict:
        """
        Compare the current window against the baseline and return a report.

        Returns a dict with:
            status      — "ok", "alert" or "insufficient_data"
            n           — number of requests in the current window
            current     — current window averages for each signal
            alerts      — signals that have drifted beyond the threshold
        """
        n = len(self.window)

        # We need at least 10 data points before the averages mean anything.
        if n < 10:
            return {
                "status": "insufficient_data",
                "n": n,
                "message": "Need at least 10 requests to report on drift.",
            }

        current = {
            "mean_length": round(statistics.mean(r["length"] for r in self.window), 2),
            "non_ascii_frac": round(
                statistics.mean(r["non_ascii_frac"] for r in self.window), 4
            ),
            "oov_rate": round(statistics.mean(r["oov_rate"] for r in self.window), 4),
        }

        # each drift check if something is wrong will be added!
        alerts: dict = {}

        # mean length drift
        baseline_mean_length = self.baseline.get("mean_length", 0.0)
        current_mean_length = current["mean_length"]
        if baseline_mean_length > 0:
            relative_shift = (
                abs(current_mean_length - baseline_mean_length) / baseline_mean_length
            )
            if relative_shift > DRIFT_THRESHOLD:
                alerts["mean_length"] = {
                    "baseline": baseline_mean_length,
                    "current": current_mean_length,
                    "shift_pct": round(relative_shift * 100, 1),
                    "threshold_pct": DRIFT_THRESHOLD * 100,
                }

        #                 baseline = 80 chars
        #                   current  = 12 chars (people sending single words!)

        #                   relative_shift = abs(12 - 80) / 80
        #                    = 68 / 80
        #                    = 0.85 (85% shift!)

        # 0.85 > 0.25 (DRIFT_THRESHOLD)
        # → ALERT! ⚠️

        # non_ascii drift
        baseline_non_ascii = self.baseline.get("non_ascii_frac", 0.0)
        current_non_ascii = current["non_ascii_frac"]
        non_ascii_delta = current_non_ascii - baseline_non_ascii
        non_ascii_threshold = max(
            baseline_non_ascii * DRIFT_THRESHOLD, MIN_NON_ASCII_ALERT_DELTA
        )
        if non_ascii_delta > non_ascii_threshold:
            alerts["non_ascii_frac"] = {
                "baseline": baseline_non_ascii,
                "current": current_non_ascii,
                "shift_pct": round(
                    (non_ascii_delta / max(baseline_non_ascii, non_ascii_threshold))
                    * 100,
                    1,
                ),
                "threshold_pct": DRIFT_THRESHOLD * 100,
            }
        # baseline = 0.003 (0.3% non-ASCII)
        # current  = 0.25  (25% non-ASCII — people sending French/Chinese!)

        # delta = 0.25 - 0.003 = 0.247

        # threshold = max(0.003 × 0.25, 0.01)
        #           = max(0.00075, 0.01)
        #           = 0.01  ← MIN_NON_ASCII_ALERT_DELTA

        # 0.247 > 0.01 → ALERT! ⚠️

        # OOV rate is only meaningful when we have a training vocabulary to compare
        # against. With an empty vocab every word looks unknown, which would make
        # the monitor useless in a fresh environment with no train.csv.
        if self._train_vocab and current["oov_rate"] > OOV_ALERT_THRESHOLD:
            alerts["oov_rate"] = {
                "current": current["oov_rate"],
                "threshold": OOV_ALERT_THRESHOLD,
            }

        # self._train_vocab check:
        #   Empty vocab → every word looks unknown
        #   → 100% OOV rate always → useless ❌
        #   Only check if vocab exists ✅

        # OOV_ALERT_THRESHOLD = 0.30
        # current oov_rate = 0.45 (45% unknown words)
        # 0.45 > 0.30 → ALERT! ⚠️

        return {
            "status": "alert" if alerts else "ok",
            "n": n,
            "current": current,
            "baseline": self.baseline,
            "alerts": alerts,
        }


# # Everything normal ✅
# {
#   "status": "ok",
#   "n": 87,
#   "current": {"mean_length": 79.3, "non_ascii_frac": 0.003, "oov_rate": 0.08},
#   "baseline": {"mean_length": 80.0, "non_ascii_frac": 0.01},
#   "alerts": {}
# }

# # Drift detected ⚠️
# {
#   "status": "alert",
#   "n": 100,
#   "current": {"mean_length": 12.1, "non_ascii_frac": 0.25, "oov_rate": 0.45},
#   "baseline": {"mean_length": 80.0, "non_ascii_frac": 0.01},
#   "alerts": {
#     "mean_length": {"baseline": 80.0, "current": 12.1, "shift_pct": 84.9},
#     "non_ascii_frac": {"baseline": 0.01, "current": 0.25, "shift_pct": 247.0},
#     "oov_rate": {"current": 0.45, "threshold": 0.30}
#   }
# }


# App starts:
#   DriftMonitor created
#   → reads train.csv
#   → builds vocab set (all training words)
#   → computes baseline (mean_length, non_ascii)

# Every /predict request:
#   → run_inference() → prediction made
#   → monitor.record(text) → stats stored in window

# Every /drift request:
#   → monitor.report()
#   → compare window vs baseline
#   → return "ok" or "alert" with details
