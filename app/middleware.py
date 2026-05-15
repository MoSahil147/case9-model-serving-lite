# middleware code sits between every incoming HTTPS requests and actual APU endpoint!
# every single request passes through it automatically!
"""
Request and response logging middleware.

Every request that passes through the API gets a structured JSON log line
written to stdout. One line, one request, everything you need to debug a
bad prediction a week later.

Why stdout and not a file?
    - HuggingFace Spaces captures stdout and makes it available in the
      container logs tab with timestamps and search.
    - In a real production setup you would ship stdout to Loki or CloudWatch
      via a log collector sidecar. Stdout is the universal sink.

Log format: one JSON object per line (JSON Lines / NDJSON).
Query with: cat logs/api.log | jq 'select(.status == 500)'
"""

import json
import logging
import time
import uuid

# starlette is the framework upon which the FastAPI is built on
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger("api.access")


def configure_logging() -> None:
    """
    Set up a clean stdout handler that emits raw JSON.

    Call this once from app startup. We do NOT use a JSON logging library
    because we want zero extra dependencies for something this simple.
    """
    handler = logging.StreamHandler() # creates the handler which writes to stdout
    # The formatter just passes the message through. We build the full JSON
    # string ourselves in the middleware so we have complete control.
    handler.setFormatter(logging.Formatter("%(message)s")) # outputting only messages nothing else

    root = logging.getLogger() # gets the root logger, the top level logger that all other logger inherit from
    root.addHandler(handler)
    root.setLevel(logging.INFO) # info


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Attaches to every inbound request and writes one JSON log line on the way out.

    We log both the request and the response body (truncated to 500 characters)
    so that anyone reading the logs can see exactly what was sent and what was
    returned without needing to re-run the inference.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4()) # generate random unique ID
        start_time = time.perf_counter() # captures current time in high resolution

        # Read and cache the raw request body.
        # BaseHTTPMiddleware re-populates request._body after we read it,
        # so the actual endpoint handler still receives the full body.
        raw_body = await request.body()

        response = await call_next(request) # await is used beacuse this is async fuction

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Collect the response body chunks so we can log them.
        # We must rebuild the response afterwards because body_iterator
        # can only be consumed once.
        response_chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            response_chunks.append(chunk)
        response_body = b"".join(response_chunks)

        log_record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), # time in ISO format
            "request_id": request_id, # UUID
            "method": request.method, # GET, POST
            "path": str(request.url.path), #/predict or /health
            "status": response.status_code, # 200, 500, etc status code
            "duration_ms": duration_ms, # how long whole request took
            # Truncate long inputs so the log file does not balloon in size.
            # 500 characters is enough to reproduce any realistic prediction.
            "input_text": raw_body.decode("utf-8", errors="replace")[:500],
            "input_length_bytes": len(raw_body),
            "response_snippet": response_body.decode("utf-8", errors="replace")[:500], # first 500 chars of response enough to see the prediction 
            "client_ip": request.client.host if request.client else None, #IP address of who sends the request
            "user_agent": request.headers.get("user-agent", ""), # broweswer of the HTTP client
        }

        logger.info(json.dumps(log_record))

        # Reconstruct the response with the body we already consumed.
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
