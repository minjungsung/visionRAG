"""Prometheus metrics for FastAPI."""
import time

from prometheus_client import Counter, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

APP_INFO = Info("visionrag", "VisionRAG application info")
APP_INFO.info({"version": "0.1.0"})

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RAG_GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "RAG generation latency",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

INGESTION_COUNT = Counter(
    "ingestion_documents_total",
    "Total documents ingested",
    ["status"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
        return response


def metrics_endpoint(request: Request) -> Response:
    """Endpoint to expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
