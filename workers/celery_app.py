"""Celery 앱 설정."""

from celery import Celery

from src.config.settings import settings

app = Celery(
    "visionrag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Reliability
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Timeouts (seconds)
    task_soft_time_limit=300,
    task_time_limit=600,
)

app.autodiscover_tasks(["workers.tasks"])
