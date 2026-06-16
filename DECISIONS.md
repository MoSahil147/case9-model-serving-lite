# Decisions Log — Case 9: Model Serving Lite

## Assumptions I made

1. English-only inputs are the intended use case — because the base model was trained on English text and inputs in other languages produce unreliable confidence scores. Rather than blocking non-English input at the API level I treat non-ASCII character frequency as a drift signal instead.

2. The base pretrained model is good enough without fine-tuning for the initial deployment — because `distilbert-base-uncased-finetuned-sst-2-english` already achieves 91-93% accuracy on English sentiment tasks out of the box. Fine-tuning is wired up in the CI pipeline for when new labelled data arrives but the first deployment does not require it.

3. A single uvicorn worker is sufficient for a free-tier prototype — because the model is a module-level singleton and running multiple workers would require each process to hold a separate 270 MB copy of the model in RAM. A demo with occasional traffic does not benefit from the added complexity.

4. In-memory drift monitoring state is acceptable for the demo — because the important thing is the architecture being correct and the extension path being obvious. A production system would write signals to a time-series database so they survive restarts and aggregate across workers.

5. The held-out evaluation set (data/eval.csv) is a fair proxy for production traffic — because for demonstrating the CI gate a clean human-readable eval set is more useful than a noisier but more realistic one. Twenty labelled examples is enough to show the gate working.

---

## Trade-offs

| Choice | Alternative | Why I picked this |
|---|---|---|
| DistilBERT (66M params) | BERT-base (110M params) | 40% smaller and 60% faster on CPU with less than 3% accuracy loss. On a free-tier host with no GPU this matters. |
| HuggingFace Spaces | Render or Fly.io | Docker SDK gives automatic HTTPS and CI-triggered redeploy with zero Nginx configuration. The natural home for an ML demo. |
| Single uvicorn worker | Gunicorn with multiple workers | See assumption 3. Simplicity wins for a one-day prototype. |
| In-memory drift window | Prometheus and Grafana | Adding a metrics sidecar would triple the infrastructure complexity for marginal benefit in a one-day project. |
| run_in_executor for inference | Async torch (does not exist) | PyTorch inference is synchronous and blocks the GIL. run_in_executor moves it to a thread pool so the event loop stays responsive during a slow CPU inference. |
| Baking model weights into the Docker image | Download on first request | Avoids 30-60 second cold starts when HF Spaces wakes the container after sleeping. Adds ~270 MB to the image but this is a one-time build cost. |
| ruff for linting and formatting | flake8 and black separately | ruff replaces both tools in one binary and runs significantly faster in CI. |
| uv for dependency management | pip + requirements.txt | uv is 10-100x faster, uses a lockfile (`uv.lock`) so every environment is reproducible, and manages Python itself in CI so there is no dependency on what the runner has pre-installed. |
| `device_map="auto"` for inference | `device=-1` (hardcoded CPU) | Auto-selects the best available device (CPU/GPU/MPS) without code changes. If a GPU becomes available the same code uses it automatically. |
| `re.findall(r'\b\w+\b')` for OOV tokenisation | `str.split()` | `split()` leaves punctuation attached to words (`"product!"` ≠ `"product"`), inflating OOV rates artificially. The regex strips punctuation at word boundaries so baseline and live request tokenisation are consistent. |
| Pinned base image `python:3.11.12-slim` | Floating `python:3.11-slim` | Floating tags silently pick up new patch releases and can introduce breaking changes in CI. Pinning the exact patch version makes builds fully reproducible. |
| Retry loop with exponential backoff for model download | Single download attempt | HuggingFace Hub returns HTTP 429 (rate limit) on shared GitHub Actions runners. A 5-attempt loop with 1/2/4/8s backoff handles transient rate limits without failing the build. |

---

## What I de-scoped and why

- GPU support — the free tier has no GPU and DistilBERT runs acceptably on two vCPUs at 130-200 ms. `device_map="auto"` means a GPU will be used automatically if one becomes available without any code change.
- Authentication on /predict — adding an API key adds friction to the demo where the judge calls the endpoint with a bare curl command. Noted as a next step in the README.
- Persistent log storage — logs go to stdout and are visible in the HF Spaces container logs. Shipping to S3 or Loki is the right production move and is noted.
- Shadow deployment — routing 5% of traffic to a candidate model alongside production is architecturally interesting but takes more time than a single day allows.
- Statistical significance tests on the metrics gate — the gate uses a fixed numeric threshold rather than a bootstrap confidence interval. For 20 eval examples the simpler approach is easier to reason about.
- DistilBERT tokenizer for OOV — using the model's actual WordPiece tokenizer for the OOV drift signal would be more accurate but couples `drift.py` to the model weights. A regex tokenizer (`re.findall`) gives consistent baseline vs. live comparison without the coupling.

---

## What I'd do differently with another day

- Add a `/metrics` endpoint in Prometheus format so Grafana can visualise request rate, p50/p99 latency and drift signals on one dashboard.
- Store drift signals in Redis so they survive container restarts and aggregate across workers.
- Write a Locust load test to find the breaking point under sustained traffic.
- Implement the shadow deployment stretch goal: route 5% of live traffic to the candidate model and compare predictions before promoting.
- Add proper API key authentication with key rotation and a revocation endpoint.
