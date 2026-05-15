# bigger picture caller sends text, PredictRequest,
# API responds back with label + score (PredictResponse),
# health check-> HealthResponse
"""
Request and response shapes for the sentiment API.

Keeping these in a separate file means the main app and the tests
can both import them without pulling in FastAPI or torch.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """What the caller sends us."""

    # We enforce a minimum length so the model never sees an empty string,
    # and a maximum length so we don't silently truncate a multi-page document
    # and return a misleading score.
    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="The text to classify as POSITIVE or NEGATIVE.",
    )


class PredictResponse(BaseModel):
    """What we send back to the caller."""

    label: str  # postive or negative
    # Score is the softmax probability for the winning label — not a raw logit.
    # 0.99 means the model is very confident; 0.51 means it is basically guessing.
    score: float
    # the git SHA gets baked into the Docker Image at built time, if if there is any prediction that goes wrong you can immediately chcek out the exact commit and reproduce! Thats called traceability
    model_version: str  # A git SHA baked into the Docker image lets you reproduce any prediction exactly
    # End-to-end latency in milliseconds, measured inside the endpoint handler.
    # Useful for spotting when a new model is significantly slower than the old one.
    latency_ms: float  # how long


class HealthResponse(BaseModel):
    """Simple liveness response."""

    status: str
