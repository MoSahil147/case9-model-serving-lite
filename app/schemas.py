# bigger picture caller sends text, PredictRequest,
# API responds back with label + score (PredictResponse),
# health check-> HealthResponse

# Someone sends text to your API
#         ↓
# API predicts sentiment
#         ↓
# API sends back label + score
#
# This file defines EXACTLY:
# → what the incoming data must look like, text field (1-2048 chars)
# → what the outgoing data will look like, label, score, version, latency

# Caller must send THIS → PredictRequest
# API will return THIS  → PredictResponse
# Health check returns  → HealthResponse

"""
Request and response shapes for the sentiment API.

Keeping these in a separate file means the main app and the tests
can both import them without pulling in FastAPI or torch.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# what the caller sends!
class PredictRequest(BaseModel):
    """What the caller sends us."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2048,
            description="The text to classify as POSITIVE or NEGATIVE.",
        ),
    ]

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class PredictResponse(BaseModel):
    """What we send back to the caller."""

    # Pydantic v2 reserves the word "model_" as a protected namespace, it use for model_fields, model_validate, model_dump etc.
    # Without this config: model_version → WARNING: conflicts with protected namespace ⚠️
    # With this config: protected_namespaces=() → empty = no protected namespaces model_version → works fine ✅

    model_config = ConfigDict(protected_namespaces=())

    label: str  # postive or negative
    # Score is the softmax probability for the winning label — not a raw logit.
    # 0.99 means the model is very confident; 0.51 means it is basically guessing.
    score: float
    # the git SHA gets baked into the Docker Image at built time, if if there is any prediction that goes wrong you can immediately chcek out the exact commit and reproduce! Thats called traceability
    # in case anything fails or goes wrong in production, reproduce the exact prediction!
    model_version: str  # A git SHA baked into the Docker image lets you reproduce any prediction exactly
    # End-to-end latency in milliseconds, measured inside the endpoint handler.
    # Useful for spotting when a new model is significantly slower than the old one.
    latency_ms: float  # how long


class HealthResponse(BaseModel):
    """Simple liveness response."""

    status: Literal["ok"]


class DriftResponse(BaseModel):
    """Response shape for the /drift endpoint."""

    status: Literal["ok", "alert", "insufficient_data"]
    n: int
    current: dict[str, float] | None = None
    baseline: dict[str, float] | None = None
    alerts: dict[str, Any] | None = None
    message: str | None = None
