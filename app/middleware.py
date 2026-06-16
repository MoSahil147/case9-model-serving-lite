# middleware code sits between every incoming HTTPS requests and actual APU endpoint!
# every single request passes through it automatically!
# one job, log every request as JSON line
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
from datetime import datetime, timezone

# starlette is the framework upon which the FastAPI is built on
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger("api.access")

# sets the logging sysy once at startup
def configure_logging() -> None:
    """
    Set up a clean stdout handler that emits raw JSON.

    Call this once from app startup. We do NOT use a JSON logging library
    because we want zero extra dependencies for something this simple.
    """
    
    # the handler that write to STDOUT
    handler = logging.StreamHandler()  # creates the handler which writes to stdout
    # The formatter just passes the message through. We build the full JSON
    # string ourselves in the middleware so we have complete control.
    
    # controls what gets printed
    handler.setFormatter(
        logging.Formatter("%(message)s")
    )  # outputting only messages nothing else

    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)


# class that wraps every request
class LoggingMiddleware(BaseHTTPMiddleware): # inherits from the BaseHTTPMiddleware gets all the middleware powers automoatically
    """
    Attaches to every inbound request and writes one JSON log line on the way out.

    We log both the request and the response body (truncated to 500 characters)
    so that anyone reading the logs can see exactly what was sent and what was
    returned without needing to re-run the inference.
    """

    # thsi inherits everything from the BaseHTTP thing, but need to implement disptach() method
    async def dispatch(self, request: Request, call_next) -> Response:
        # request is the incoming HTTPS request
        # call_next is the function to pass request to actual endpoint, its like i have done my preporcessing, now passing the request to real endpoint and give me back the response
        # Response is the returning the HTTP response
        
        request_id = str(uuid.uuid4())  # generate random unique ID
        start_time = time.perf_counter()  # captures current time in high resolution

        # Read and cache the raw request body.
        # BaseHTTPMiddleware re-populates request._body after we read it,
        # so the actual endpoint handler still receives the full body.
        # BaseHTTP auto repopulates, endpoit reqcieves the full body, without this, endpoint will get empty
        
        # read what the user sent, without this endpoint will recive nothing
        raw_body = await request.body()

        # passes request to the real API endpoint (like /predict or /healthz)
        # await=waits for response without blocking, other req can be handled meanwhile
        
        # passing to the actual endpoint
        # call_next = "go run the real endpoint now, give me back the response." 
        # await means it waits for the result without freezing the server — other requests can be handled meanwhile.
        response = await call_next(
            request
        )  # await is used beacuse this is async function

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Collect the response body chunks so we can log them.
        # We must rebuild the response afterwards because body_iterator
        # can only be consumed once.
        
        # response body arrives as CHUNKS, can only be read ONCE, without collecting, caller gets empty response
        # we need this, beacuse without this we will get empty response body
        response_chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            response_chunks.append(chunk)
        response_body = b"".join(response_chunks)

        log_record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": request_id,  # UUID
            "method": request.method,  # GET, POST, GET is to get data from the server, POST is to send data to server
            "path": str(request.url.path),  # /predict or /health
            "status": response.status_code,  # 200, 500, etc status code
            "duration_ms": duration_ms,  # how long whole request took
            # Truncate long inputs so the log file does not balloon in size.
            # 500 characters is enough to reproduce any realistic prediction.
            "input_text": raw_body.decode("utf-8", errors="replace")[:500],
            "input_length_bytes": len(raw_body),
            "response_snippet": response_body.decode("utf-8", errors="replace")[
                :500
            ],  # first 500 chars of response enough to see the prediction, enough to reproduce any preduction, keeps log files manageable
            # if bytes cant be decoded as UTF-8, we will replace with ?
            "client_ip": request.client.host
            if request.client
            else None,  # IP address of who sends the request
            "user_agent": request.headers.get(
                "user-agent", ""
            ),  # broweswer of the HTTP client
        }

        # write to log, convert the dict to a single JSON line and writes it! 
        # one request = one line
        logger.info(json.dumps(log_record))

        # Reconstruct the response with the body we already consumed.
        return Response(
            content=response_body, # the body bytes we collecetd 
            status_code=response.status_code, 
            headers=dict(response.headers),
            media_type=response.media_type,
        )


# Request arrives:
# POST /predict {"text": "Amazing!"}
#         ↓
# LoggingMiddleware.dispatch():
#   → generate request_id
#   → start timer
#   → read request body
#         ↓
# call_next(request):
#   → actual /predict endpoint runs
#   → model predicts → POSITIVE 0.99
#         ↓
# Back in middleware:
#   → calculate duration_ms
#   → collect response chunks
#   → build log_record dict
#   → json.dumps → one JSON line
#   → logger.info → write to stdout
#         ↓
# Rebuild response
#         ↓
# Return to caller:
# {"label":"POSITIVE","score":0.99,"latency_ms":45.3}