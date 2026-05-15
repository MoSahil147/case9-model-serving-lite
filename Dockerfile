# Multi-stage build so we do not ship build tools into the production image.

# Stage 1 (builder): installs Python packages and downloads the model weights.
# Stage 2 (runtime): copies only what is needed to run the API.

# Why bake the model weights into the image?
#   HuggingFace Spaces puts free-tier spaces to sleep after 15 minutes of
#   inactivity. When the space wakes up the container restarts from scratch.
#   Downloading 270 MB of weights on every cold start would add 30-60 seconds
#   to the first request. Baking the weights into the layer cache reduces
#   cold-start model load to roughly 3-5 seconds (reading from disk, not network).


# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies first so Docker can cache this layer independently
# from the application code. Changing app code will not invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the model weights using the same MODEL_NAME that the app uses.
# They land in /root/.cache/huggingface and we copy that cache into the runtime stage.
ARG MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ENV MODEL_NAME=${MODEL_NAME}

RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='${MODEL_NAME}')"


# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Run as a non-root user. If the container is ever compromised, the attacker
# cannot write to system directories or install packages.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy the installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy the model cache that was downloaded in the builder stage.
# We put it in the app user's home so $HF_HOME resolves correctly.
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy the application source and the training data (needed for drift baseline).
COPY app/ ./app/
COPY data/ ./data/

# The git SHA is injected at build time by the CI pipeline.
# It appears in every /predict response under the model_version field so
# any prediction can be traced back to the exact commit that produced it.
ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA}

ARG MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ENV MODEL_NAME=${MODEL_NAME}

# Tell HuggingFace where to find the cached weights.
ENV HF_HOME=/home/appuser/.cache/huggingface

# HuggingFace Spaces expects the app to listen on port 7860.
EXPOSE 7860

USER appuser

# Single worker because the model is a module-level singleton.
# Multiple workers would each load a separate copy into RAM, exhausting
# the 16 GB free-tier allocation on a moderately sized model.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
