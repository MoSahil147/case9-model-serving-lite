---
title: Sentiment API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Case 9: Model Serving Lite

**Live demo:** `https://YOUR_HF_USERNAME-sentiment-api.hf.space` *(update after deployment)*
**Repo:** https://github.com/YOUR_GITHUB_USERNAME/case9-model-serving-lite
**Demo video:** *(add Loom/YouTube link here)*

[![CI](https://github.com/YOUR_GITHUB_USERNAME/case9-model-serving-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/case9-model-serving-lite/actions/workflows/ci.yml)

---

## What This Is

A production-grade sentiment classification service that takes a data scientist's notebook
model and turns it into a monitored deployable API with a CI pipeline that automatically
rejects retrained models that regress on quality metrics.

Built for: any team that wants to ship an NLP model to production and actually know when it
starts going wrong.

---

## Predict Endpoint — Quick Start

```bash
curl -X POST https://YOUR_HF_USERNAME-sentiment-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely brilliant and exceeded every expectation."}'
```

Response:

```json
{
  "label": "POSITIVE",
  "score": 0.999812,
  "model_version": "sha-a1b2c3d",
  "latency_ms": 148.3
}
```

Other endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Liveness check for uptime monitors |
| `/predict` | POST | Classify text as POSITIVE or NEGATIVE |
| `/drift` | GET | Current drift monitoring report |
| `/docs` | GET | Auto-generated OpenAPI documentation |

---

## How to Run Locally

**Option A: Docker (recommended — exactly what production runs)**

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/case9-model-serving-lite.git
cd case9-model-serving-lite
docker build -t sentiment-api .
docker run -p 7860:7860 sentiment-api
```

Then open http://localhost:7860/docs

**Option B: Direct Python**

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/case9-model-serving-lite.git
cd case9-model-serving-lite
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Note: the first run will download ~270 MB of model weights. Subsequent runs load from cache.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| API framework | FastAPI | Async support, automatic OpenAPI docs and Pydantic validation out of the box |
| Model | distilbert-base-uncased-finetuned-sst-2-english | 40% smaller and 60% faster than BERT-base with less than 3% accuracy loss |
| Container | Docker (multi-stage) | Keeps build tools out of the runtime image and bakes model weights in to avoid cold-start downloads |
| Registry | ghcr.io | Free for public repos, no separate account needed and uses GITHUB_TOKEN |
| Hosting | HuggingFace Spaces (Docker SDK) | Free, automatic HTTPS and the natural home for ML model demos |
| CI | GitHub Actions | Native to the repo, generous free tier and integrates with ghcr.io without configuration |
| Logging | JSON Lines to stdout | Universal sink that HF Spaces captures automatically and that ships trivially to any log aggregator |
| Linting | ruff | Replaces flake8 and black in one tool and runs significantly faster in CI |

---

## Architecture

```
Caller
  │
  ▼
POST /predict
  │
  ├── LoggingMiddleware  ──► stdout (JSON Lines)
  │
  ├── Schema validation  ──► 422 if text is empty or > 2048 chars
  │
  ├── run_inference()    ──► ThreadPoolExecutor (keeps event loop free)
  │       │
  │       └── _do_inference()  ──► DistilBERT pipeline (CPU)
  │
  ├── DriftMonitor.record()  ──► updates in-memory rolling window
  │
  └── PredictResponse  ──► label, score, model_version, latency_ms

GET /drift
  │
  └── DriftMonitor.report()
        ├── compare window vs baseline (mean_length, non_ascii_frac)
        └── check OOV rate threshold
```

---

## CI / CD Pipeline

```
Push to main
    │
    ├── lint-and-test
    │       ├── ruff check + format
    │       └── pytest (model mocked — no download needed)
    │
    ├── build-and-push  (only on main)
    │       ├── docker build --build-arg GIT_SHA=<sha>
    │       └── push to ghcr.io with tags sha-<sha> and latest
    │
    └── deploy  (only on main)
            └── upload to HuggingFace Spaces via HF Hub API

PR touching data/train.csv
    │
    └── retrain-and-gate
            ├── guard: reject if eval.csv also changed
            ├── python scripts/retrain.py   (fine-tune for 2 epochs)
            ├── python scripts/evaluate.py  (run on held-out eval.csv)
            ├── python scripts/check_metrics.py
            │       ├── PASS  → commit new baseline.json, auto-merge PR
            │       └── FAIL  → comment on PR with details, exit 1
```

---

## How Would I Know This Model Is Failing Before Customers Do?

Three layers of detection are built into this service.

**Layer 1 — Input drift (leading indicator)**
The `/drift` endpoint monitors the statistical properties of incoming text against the
training corpus: character length, non-ASCII fraction and vocabulary novelty. A shift in
any of these signals means the model is being used differently from how it was trained.
Set up UptimeRobot (free tier) to ping `/drift` every five minutes and alert on
`"status": "alert"`. This catches problems before any customer complains.

**Layer 2 — Confidence distribution (concurrent indicator)**
Every prediction writes a structured JSON log line that includes the score field.
A healthy sentiment model is rarely uncertain — most scores should be above 0.85.
A spike in low-confidence predictions (score 0.50 to 0.70) means the model is
struggling with inputs it has not seen before.

```bash
cat logs/api.log | jq 'select(.status == 200) | .response_snippet | fromjson | select(.score < 0.75)'
```

**Layer 3 — Error rate and latency (trailing indicator)**
The logs also capture HTTP status codes and duration_ms per request. A 500-rate spike
or P99 latency jump often means a dependency conflict after a package update rather than
model degradation — but it is still a production failure.

```bash
cat logs/api.log | jq 'select(.status >= 500) | {ts, path, duration_ms}'
```

**Layer 4 — CI gate (proactive)**
No regressed model can be promoted. The retrain workflow compares every candidate model
against the frozen eval set before merging. Degradation from bad training data is blocked
at the source.

---

## What Is NOT Done

- **Authentication.** A real service would require an API key in the request header.
  Removed for the demo so judges can call the endpoint with a bare curl command.
- **Persistent drift storage.** The drift window resets on container restart. A production
  setup would push signals to Prometheus and query them in Grafana.
- **GPU support.** The free-tier host has no GPU. Response times are 130-200 ms on CPU
  which is acceptable for a demo but would need a GPU for high-traffic production use.
- **Shadow deployment.** The stretch goal of mirroring 5% of traffic to a candidate model
  is architecturally sound but the live routing logic is not wired up in this version.

## In Production I Would Also Add

- A `/metrics` endpoint in Prometheus format so Grafana can show request rate, latency
  percentiles and drift signals on a single dashboard.
- Redis-backed drift storage so signals survive restarts and aggregate across workers.
- Rate limiting (e.g., SlowAPI) to prevent a single caller from flooding the service.
- A warm-up ping from the CI deploy step so HF Spaces does not serve a cold-start request
  to the first real caller.
- Proper API key authentication with key rotation and a revocation endpoint.
