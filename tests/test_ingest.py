"""Ingestion endpoint tests with mocked pipeline."""

import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

MOCK_DOC_ID = str(uuid.uuid4())


# --- Sync Ingestion Tests ---


@pytest.mark.asyncio
async def test_ingest_returns_doc_id(client):
    """POST /ingest with a valid file returns a doc_id."""
    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest_file.return_value = MOCK_DOC_ID
    mock_pipeline_cls.return_value = mock_pipeline_instance

    with patch.dict(
        "src.ingestion.pipeline.__dict__",
        {"IngestionPipeline": mock_pipeline_cls},
        clear=False,
    ):
        pass

    # Since the endpoint uses `from src.ingestion.pipeline import IngestionPipeline`
    # inside the function body, we need to mock the module attribute
    mock_module = MagicMock()
    mock_module.IngestionPipeline = mock_pipeline_cls

    with patch.dict(sys.modules, {"src.ingestion.pipeline": mock_module}):
        with patch("src.mlops.metrics.INGESTION_COUNT") as mock_counter:
            mock_counter.labels.return_value = MagicMock()
            resp = await client.post(
                "/ingest",
                files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "doc_id" in data
    assert data["doc_id"] == MOCK_DOC_ID


@pytest.mark.asyncio
async def test_ingest_no_file_returns_422(client):
    """POST /ingest without a file returns validation error."""
    resp = await client.post("/ingest")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_calls_pipeline_with_content(client):
    """Verify the pipeline is called with file name and content bytes."""
    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest_file.return_value = MOCK_DOC_ID
    mock_pipeline_cls.return_value = mock_pipeline_instance

    mock_module = MagicMock()
    mock_module.IngestionPipeline = mock_pipeline_cls

    file_content = b"hello world document"

    with patch.dict(sys.modules, {"src.ingestion.pipeline": mock_module}):
        with patch("src.mlops.metrics.INGESTION_COUNT") as mock_counter:
            mock_counter.labels.return_value = MagicMock()
            await client.post(
                "/ingest",
                files={"file": ("document.txt", file_content, "text/plain")},
            )

    mock_pipeline_instance.ingest_file.assert_called_once_with("document.txt", file_content)


@pytest.mark.asyncio
async def test_ingest_increments_success_metric(client):
    """Verify the Prometheus success counter is incremented."""
    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest_file.return_value = MOCK_DOC_ID
    mock_pipeline_cls.return_value = mock_pipeline_instance

    mock_module = MagicMock()
    mock_module.IngestionPipeline = mock_pipeline_cls

    with patch.dict(sys.modules, {"src.ingestion.pipeline": mock_module}):
        with patch("src.mlops.metrics.INGESTION_COUNT") as mock_counter:
            mock_label = MagicMock()
            mock_counter.labels.return_value = mock_label

            await client.post(
                "/ingest",
                files={"file": ("test.pdf", b"content", "application/pdf")},
            )

            mock_counter.labels.assert_called_with(status="success")
            mock_label.inc.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_pipeline_error_propagates(client):
    """POST /ingest raises when the pipeline encounters a fatal error."""
    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest_file.side_effect = RuntimeError("Parse failed")
    mock_pipeline_cls.return_value = mock_pipeline_instance

    mock_module = MagicMock()
    mock_module.IngestionPipeline = mock_pipeline_cls

    with patch.dict(sys.modules, {"src.ingestion.pipeline": mock_module}):
        with patch("src.mlops.metrics.INGESTION_COUNT"):
            with pytest.raises(RuntimeError, match="Parse failed"):
                await client.post(
                    "/ingest",
                    files={"file": ("bad.pdf", b"corrupt", "application/pdf")},
                )


# --- Async Ingestion Tests ---


@pytest.mark.asyncio
async def test_ingest_async_returns_task_id(client):
    """POST /ingest/async returns a task_id and queued status."""
    mock_tasks_module = MagicMock()
    mock_result = MagicMock()
    mock_result.id = "task-123-abc"
    mock_tasks_module.ingest_document.delay.return_value = mock_result

    with patch.dict(sys.modules, {"workers.tasks": mock_tasks_module}):
        resp = await client.post(
            "/ingest/async",
            files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "task-123-abc"
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_ingest_async_no_file_returns_422(client):
    """POST /ingest/async without a file returns validation error."""
    resp = await client.post("/ingest/async")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_async_passes_hex_content(client):
    """Verify async ingestion passes file content as hex string to Celery task."""
    mock_tasks_module = MagicMock()
    mock_result = MagicMock()
    mock_result.id = "task-456"
    mock_tasks_module.ingest_document.delay.return_value = mock_result

    file_content = b"test document bytes"

    with patch.dict(sys.modules, {"workers.tasks": mock_tasks_module}):
        await client.post(
            "/ingest/async",
            files={"file": ("doc.pdf", file_content, "application/pdf")},
        )

    mock_tasks_module.ingest_document.delay.assert_called_once_with("doc.pdf", file_content.hex())


# --- Task Status Tests ---


@pytest.mark.asyncio
async def test_get_task_status_pending(client):
    """GET /tasks/{task_id} returns current task state."""
    mock_celery_module = MagicMock()
    mock_async_result = MagicMock()
    mock_async_result.state = "PENDING"
    mock_async_result.result = None
    mock_celery_module.app.AsyncResult.return_value = mock_async_result

    with patch.dict(sys.modules, {"workers.celery_app": mock_celery_module}):
        resp = await client.get("/tasks/task-123-abc")

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "task-123-abc"
    assert data["state"] == "PENDING"
    assert data["result"] is None


@pytest.mark.asyncio
async def test_get_task_status_success(client):
    """GET /tasks/{task_id} returns completed task with result."""
    mock_celery_module = MagicMock()
    mock_async_result = MagicMock()
    mock_async_result.state = "SUCCESS"
    mock_async_result.result = {"doc_id": MOCK_DOC_ID}
    mock_celery_module.app.AsyncResult.return_value = mock_async_result

    with patch.dict(sys.modules, {"workers.celery_app": mock_celery_module}):
        resp = await client.get("/tasks/task-456-def")

    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "SUCCESS"
    assert data["result"] == {"doc_id": MOCK_DOC_ID}


@pytest.mark.asyncio
async def test_get_task_status_failure(client):
    """GET /tasks/{task_id} returns failed task state."""
    mock_celery_module = MagicMock()
    mock_async_result = MagicMock()
    mock_async_result.state = "FAILURE"
    mock_async_result.result = "Error: file parsing failed"
    mock_celery_module.app.AsyncResult.return_value = mock_async_result

    with patch.dict(sys.modules, {"workers.celery_app": mock_celery_module}):
        resp = await client.get("/tasks/task-789-ghi")

    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "FAILURE"
    assert "Error" in str(data["result"])
