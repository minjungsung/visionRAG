"""Celery 앱 설정."""
from celery import Celery

app = Celery("visionrag", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
