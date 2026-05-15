# Submission Notes — Case 9: Model Serving Lite

## What the Judges Should Know First

The most important part of this submission is not the API itself. The differentiator is the retrain workflow in `.github/workflows/retrain.yml`. It demonstrates a complete MLOps feedback loop: labelled data arrives in a PR, the model retrains automatically, the gate compares metrics against the committed baseline and the PR either merges itself or receives a clear rejection comment explaining why it failed.

The writeup answer to "how would I know this model is failing before customers do" is in the README under the same heading. Four layers: input drift, confidence distribution, error rate/latency and the CI gate. The drift endpoint is the one worth clicking during evaluation.

---

## Live Endpoints

```
Space:    https://huggingface.co/spaces/sahil147/sentiment-api
Health:   https://sahil147-sentiment-api.hf.space/healthz
Predict:  https://sahil147-sentiment-api.hf.space/predict
Drift:    https://sahil147-sentiment-api.hf.space/drift
API Docs: https://sahil147-sentiment-api.hf.space/docs
```

Example curl (no setup needed):

```bash
curl -X POST https://sahil147-sentiment-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely brilliant and exceeded every expectation."}'
```

Expected response:

```json
{
  "label": "POSITIVE",
  "score": 0.999812,
  "model_version": "sha-abc1234",
  "latency_ms": 148.3
}
```

---

## Known Limitations

- **Cold start.** HF Spaces free tier sleeps after 15 minutes of inactivity. The first request after a sleep takes 5-10 seconds. Subsequent requests are 130-200 ms. A production deployment would use a warm-up ping.
- **Drift window resets on restart.** The drift monitor is in-memory. A new deployment clears the window and the status returns to `insufficient_data` until 10 requests arrive.
- **Eval set is small.** Twenty examples is enough to demonstrate the gate but not enough for statistically meaningful estimates.

---

## Demo Video Suggested Order

1. Show `/healthz` returning 200 in the browser.
2. Run the predict curl and show label, score and model_version.
3. Open the HF Spaces logs tab and show the JSON log line that was written.
4. Simulate drift by sending Spanish text repeatedly and hit `/drift` to show the alert on `non_ascii_frac`.
5. Show a bad-data PR triggering the retrain workflow and failing with a comment.
6. Show a good-data PR passing and auto-merging.

---

## Quick Drift Simulation

Run this locally after the service is live:

```python
import requests

BASE_URL = "https://sahil147-sentiment-api.hf.space"
SPANISH = "Esto es una reseña en español: me encanta este producto, es fantástico."

for _ in range(10):
    requests.post(f"{BASE_URL}/predict", json={"text": "Great product, arrived quickly."})

for _ in range(100):
    requests.post(f"{BASE_URL}/predict", json={"text": SPANISH})

print(requests.get(f"{BASE_URL}/drift").json())
```
