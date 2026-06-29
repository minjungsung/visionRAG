"""LangSmith tracing for RAG pipeline."""
from __future__ import annotations

import functools
import time
from typing import Any

from src.config.settings import settings

_client = None


def get_langsmith_client():
    """Lazy-init LangSmith client."""
    global _client
    if _client is None and settings.langsmith_tracing:
        from langsmith import Client

        _client = Client(api_key=settings.langsmith_api_key)
    return _client


def traceable(run_type: str = "chain", name: str | None = None):
    """Decorator to trace a function with LangSmith."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not settings.langsmith_tracing:
                return fn(*args, **kwargs)

            from langsmith.run_helpers import traceable as ls_traceable

            traced_fn = ls_traceable(
                run_type=run_type,
                name=name or fn.__name__,
                project_name=settings.langsmith_project,
            )(fn)
            return traced_fn(*args, **kwargs)

        return wrapper

    return decorator


def trace_retrieval(query: str, results: list[dict], latency_ms: float) -> None:
    """Log a retrieval step to LangSmith."""
    client = get_langsmith_client()
    if client is None:
        return

    client.create_run(
        name="retrieval",
        run_type="retriever",
        inputs={"query": query},
        outputs={"documents": results},
        extra={"latency_ms": latency_ms, "num_results": len(results)},
        project_name=settings.langsmith_project,
    )


def trace_generation(query: str, context: str, answer: str, latency_ms: float) -> None:
    """Log a generation step to LangSmith."""
    client = get_langsmith_client()
    if client is None:
        return

    client.create_run(
        name="generation",
        run_type="llm",
        inputs={"query": query, "context": context},
        outputs={"answer": answer},
        extra={"latency_ms": latency_ms},
        project_name=settings.langsmith_project,
    )


def trace_rag_pipeline(query: str, results: list[dict], answer: str, timings: dict) -> None:
    """Log a full RAG pipeline run as a parent trace."""
    client = get_langsmith_client()
    if client is None:
        return

    client.create_run(
        name="rag_pipeline",
        run_type="chain",
        inputs={"query": query},
        outputs={"answer": answer, "num_sources": len(results)},
        extra={"timings": timings},
        project_name=settings.langsmith_project,
    )
