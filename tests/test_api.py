"""
Integration tests for the HTTP API layer.

These tests exercise the full request/response cycle: middleware, endpoint
logic, schema validation and error handling. The model itself is mocked
via the autouse fixture in conftest.py.
"""

def test_health_returns_ok(client):
    """The health endpoint should always return 200 with status ok."""
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_shape(client):
    """
    A well-formed request should return all four fields in the response.
    We do not assert the exact label because the mock can be changed by
    other tests; we just check the shape is correct.
    """
    response = client.post("/predict", json={"text": "This product is absolutely brilliant."})

    assert response.status_code == 200

    body = response.json()
    assert "label" in body
    assert "score" in body
    assert "model_version" in body
    assert "latency_ms" in body


def test_predict_label_is_positive_or_negative(client):
    """The label field must always be one of the two known classes."""
    response = client.post("/predict", json={"text": "I love this so much."})

    label = response.json()["label"]
    assert label in ("POSITIVE", "NEGATIVE"), f"Unexpected label: {label}"


def test_predict_score_is_between_zero_and_one(client):
    """Score is a softmax probability and must live in [0.0, 1.0]."""
    response = client.post("/predict", json={"text": "This is fine."})

    score = response.json()["score"]
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_predict_latency_ms_is_positive(client):
    """Latency must be a positive number; a zero or negative value means the clock is broken."""
    response = client.post("/predict", json={"text": "Testing latency."})

    latency = response.json()["latency_ms"]
    assert latency > 0, f"Latency should be positive but got: {latency}"


def test_predict_empty_text_returns_422(client):
    """An empty string violates the min_length=1 constraint and must be rejected."""
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_missing_text_field_returns_422(client):
    """A request body with no text field at all should also fail validation."""
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_text_too_long_returns_422(client):
    """Text longer than 2048 characters must be rejected before hitting the model."""
    long_text = "a" * 2049
    response = client.post("/predict", json={"text": long_text})

    assert response.status_code == 422


def test_drift_endpoint_returns_200(client):
    """The /drift endpoint should always be reachable."""
    response = client.get("/drift")

    assert response.status_code == 200


def test_drift_reports_insufficient_data_with_empty_window(client):
    """
    With a fresh window and no requests yet the drift endpoint should
    return insufficient_data, not ok or alert.
    """
    response = client.get("/drift")

    assert response.json()["status"] == "insufficient_data"


def test_drift_status_becomes_ok_after_enough_requests(client):
    """
    After 10 or more normal English requests the drift status should be ok.
    The mock returns POSITIVE 0.9998 every time which is within normal range.
    """
    for _ in range(12):
        client.post("/predict", json={"text": "This is a perfectly normal English sentence about product quality."})

    report = client.get("/drift").json()
    assert report["status"] == "ok"
    assert report["n"] >= 10


def test_drift_detects_non_ascii_spike(client):
    """
    Sending a lot of requests with high non-ASCII content should trigger
    a drift alert on the non_ascii_frac signal.
    """
    # First fill the window with normal requests so we have a baseline comparison.
    for _ in range(10):
        client.post("/predict", json={"text": "Normal English sentence here."})

    # Now flood with non-ASCII text simulating a language shift.
    non_english = "Esto es una reseña en español con muchos caracteres especiales: áéíóú ñ ü"
    for _ in range(100):
        client.post("/predict", json={"text": non_english})

    report = client.get("/drift").json()
    assert report["status"] == "alert"
    assert "non_ascii_frac" in report["alerts"]
