# this will load the AI model and runs the prediction! Two core responsibilities, load once and run efficiently!

# inference means feeding the input to a trained model and getting the prediction
# app starts load_model(),  _classifier lives in memory
# request comes in,  run_inference(), _do_inference(), result

# 2 core jobs, load the AI model once and run the predictions efficiently
"""
Model loading and inference.

The classifier is loaded once when the app starts and kept in memory for
the lifetime of the process.
Loading it on every request would add two to three seconds of cold-start latency per call which is completely unacceptable.

We use a module-level variable rather than threading.local() because we run
with a single uvicorn worker — there is only ever one process and one model.
"""

# we choosed module level as there is single variable shared by everyone, where as threading.local() stroes a separte copy for each thread
import asyncio  # helps to run things concurrently, allows python to do multiple things without waiting for each one to finish
import os
from typing import Any

from transformers import (
    pipeline,
)  # imported pipeline from transformers library, easy wrapper around the complex models!
# converts text to tokens manually, runs model manually, converts output manually

# The model name can be overridden via environment variable so we can
# swap in a smaller model for tests without changing any code.
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

# global vraibale that will hold the loaded model, stars as None, underscore means its private
# if someone will call the inference before loading the model, it crahes which is intentinal, that is the real bug
_classifier = None


def load_model() -> None:  # None as the function doenst return anything
    """
    Download (or load from cache) the sentiment model and store it globally.
    Called once from the FastAPI lifespan context manager.
    """
    global _classifier  # tells python we are modifying the module-level varibale, not creating a new local one

    print(f"Loading model: {MODEL_NAME}")
    # device=-1 forces CPU inference. We have no GPU on the free-tier host.
    _classifier = pipeline("sentiment-analysis", model=MODEL_NAME, device_map="auto")
    print("Model loaded and ready.")
    # it is called once from the FastAPI lifespan context manager! By the time first /predict request arrives the model is already sitting in memory and is ready to go!


def _do_inference(text: str) -> dict[str, Any]:
    # blocking means while the function is running, everything else waits! donenst gives control back until done
    # GIL is Golbal Interpreter Lock, Python's rule that says only one thing can run at a time! PyTorch grabs this lock while doing the inferece and holds until done!
    # run_in_executor = "go run this blocking function in a background thread, don't do it here"
    """
    The actual blocking call to the model.

    This is a plain synchronous function because torch inference blocks the
    GIL.
    We call it via run_in_executor so uvicorn's event loop stays free
    to answer health checks while a slow CPU inference is running.
    """
    # truncation=True silently cuts text longer than 512 tokens.
    # We already cap input at 2048 characters in the schema but a single
    # character can be multiple tokens so we still need this guard.

    # _classifier is our loaded model sitting in memory, we call it like function passing the text
    result = _classifier(text, truncation=True, max_length=512)  # type: ignore[misc], claasifier is the loaded model sitting in memory loaded at startup, if text too long truncation=True silently cuts it at the limit, token limit is 512
    # truncation=True, silently cuts at the limit instead of crashing
    return result[
        0
    ]  # pipeline returns a list, clean dict; we always send one text at a time


# async is non blocking function, it can pause and let other things run while waiting
# no underscore this is what API actually calls!
# takes text to analyse and return dictonary!
# "wrapper" — this function doesn't do the real work itself, it just wraps _do_inference and calls it safely
async def run_inference(text: str) -> dict[str, Any]:
    """
    Non-blocking wrapper around _do_inference.

    By running the sync function in the default thread-pool executor we
    avoid blocking the event loop, which keeps /healthz responsive even
    while /predict is working through a slow inference.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _do_inference, text)
    return result
