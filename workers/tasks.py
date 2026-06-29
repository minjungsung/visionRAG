"""비동기 인제스천 태스크."""
from workers.celery_app import app
from src.ingestion.pipeline import IngestionPipeline


@app.task(bind=True)
def ingest_document(self, file_path: str, file_bytes_hex: str):
    """파일을 비동기로 인제스천."""
    self.update_state(state="PROCESSING")
    pipeline = IngestionPipeline()
    doc_id = pipeline.ingest_file(file_path, bytes.fromhex(file_bytes_hex))
    return {"doc_id": doc_id, "status": "completed"}
