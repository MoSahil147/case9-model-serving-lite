# entry point of the web service, manages the entire lifecycle of the applictaion from loading the AI model inro ememory to server starting
"""
Sentiment API — entry point.

Three endpoints:

    GET  /healthz   — liveness check for load balancers and uptime monitors
    POST /predict   — run sentiment classification on a piece of text
    GET  /drift     — return the current drift monitoring report

The model is loaded once during startup using FastAPI's lifespan context
manager. All state that must persist across requests (the drift monitor)
lives on app.state so it is easy to access and easy to mock in tests.
"""

import os
import time
from contextlib import asynccontextmanager # allows lifespan function to pause and resume between server startup and shutdown

from fastapi import FastAPI, Request

from app.drift import DriftMonitor
from app.middleware import LoggingMiddleware, configure_logging
from app.predict import load_model, run_inference
from app.schemas import HealthResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before the `yield` runs once on startup.
    Code after the `yield` runs once on shutdown.

    We use this instead of the deprecated @app.on_event("startup") decorator.
    """
    configure_logging()
    load_model() # loading the heavy DistilBERT model
    app.state.drift = DriftMonitor() # initialise the monitring thing
    yield # server is running
    # Nothing to clean up on shutdown — torch releases GPU memory automatically.


app = FastAPI(
    title="Sentiment API",
    description="Classifies text as POSITIVE or NEGATIVE using DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)

# The logging middleware wraps every request, including /healthz.
# This is intentional: we want to see health check traffic in the logs
# so we can spot when an uptime monitor is hitting us at an unexpected rate.
app.add_middleware(LoggingMiddleware)


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def health_check():
    """
    Liveness endpoint.

    Returns 200 as long as the process is running. We deliberately do NOT
    test whether the model is loaded here because a failing model load would
    crash the startup phase, so if this endpoint is reachable the model is fine.
    """
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request_body: PredictRequest, request: Request):
    """
    Run sentiment classification on a piece of text.

    The model returns a label (POSITIVE or NEGATIVE) and a confidence score.
    We also return the model version (git SHA) and the inference latency so
    callers can track these over time without digging into logs.
    """
    start = time.perf_counter()

    # run_inference is async so it does not block the event loop.
    result = await run_inference(request_body.text)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # Tell the drift monitor about this request so it can update its
    # rolling statistics. We do this after inference so a slow model does
    # not slow down the monitoring path.
    request.app.state.drift.record(request_body.text)

    return PredictResponse(
        label=result["label"],
        score=round(result["score"], 6),
        model_version=os.getenv("GIT_SHA", "dev"),
        latency_ms=latency_ms,
    )


@app.get("/drift", tags=["ops"])
def drift_report(request: Request):
    """
    Return the current drift monitoring report.

    Useful to hit from an uptime monitor or a scheduled job. If the status
    is "alert" something about the incoming traffic has changed meaningfully
    relative to what the model was trained on.
    """
    return request.app.state.drift.report()
