"""
Unit tests for the inference layer.

We test that run_inference correctly calls the classifier singleton and
returns the expected shape. The classifier is mocked globally by conftest.py.
"""

import pytest


@pytest.mark.asyncio
async def test_run_inference_returns_label_and_score(mock_pipeline):
    """
    run_inference should return a dict with label and score keys.
    The mock pipeline is already configured to return POSITIVE 0.9998.
    """
    # We need to call load_model() so _classifier is set to the mock.
    import app.predict as predict_module

    predict_module.load_model()

    from app.predict import run_inference

    result = await run_inference("This is a great product.")

    assert "label" in result
    assert "score" in result
    assert result["label"] == "POSITIVE"
    assert result["score"] == pytest.approx(0.9998)


@pytest.mark.asyncio
async def test_run_inference_passes_text_to_classifier(mock_pipeline):
    """
    The text we pass to run_inference should be forwarded to the classifier.
    This ensures no silent text transformation happens between input and model.
    """
    import app.predict as predict_module

    predict_module.load_model()

    from app.predict import run_inference

    await run_inference("Check this exact string please.")

    # mock_pipeline is the mock classifier instance itself.
    # Calling it with a string and keyword args is what _do_inference does.
    mock_pipeline.assert_called_once()
    call_args = mock_pipeline.call_args
    assert "Check this exact string please." in call_args[0]


def test_load_model_sets_classifier(mock_pipeline):
    """
    After load_model() is called _classifier should no longer be None.
    This is the most basic possible sanity check for the startup sequence.
    """
    import app.predict as predict_module

    predict_module._classifier = None  # reset to simulate a fresh process
    predict_module.load_model()

    assert predict_module._classifier is not None
