"""
Shared test fixtures.

The most important thing here is the mock_pipeline fixture which patches
the HuggingFace pipeline() function before any test runs. Without this,
every test would trigger a ~270 MB model download and a slow load, which
makes CI take minutes and local test runs painful.

The patch target is "app.predict.pipeline" because that is the name as it
exists inside the predict module at the time of the call, not the name
inside the transformers library. This is the standard Python mock convention.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_pipeline():
    """
    Replace the real HuggingFace pipeline with a lightweight mock for every test.

    The mock classifier returns a fixed positive prediction by default.
    Individual tests can override this by calling mock_clf.return_value = [...].
    """
    mock_clf = MagicMock()
    # The real pipeline returns a list of dicts; our code always uses result[0].
    mock_clf.return_value = [{"label": "POSITIVE", "score": 0.9998}]

    with patch("app.predict.pipeline", return_value=mock_clf):
        yield mock_clf


@pytest.fixture()
def client(mock_pipeline):
    """
    A FastAPI TestClient with the full app lifespan running.

    Using `with TestClient(app) as c` triggers the startup and shutdown events
    so load_model() and DriftMonitor.__init__() both run, exactly as they
    would in production. The model is mocked by the autouse fixture above.
    """
    from app.main import app

    with TestClient(app) as c:
        yield c
