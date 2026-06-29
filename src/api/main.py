"""VisionRAG API 서버."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.config.settings import settings
from src.mlops.metrics import PrometheusMiddleware, metrics_endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="VisionRAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics_endpoint)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    from src.ingestion.pipeline import IngestionPipeline
    from src.mlops.metrics import INGESTION_COUNT

    pipeline = IngestionPipeline()
    content = await file.read()
    doc_id = pipeline.ingest_file(file.filename, content)
    INGESTION_COUNT.labels(status="success").inc()
    return {"doc_id": doc_id}


@app.post("/ingest/async")
async def ingest_async(file: UploadFile = File(...)):
    from workers.tasks import ingest_document

    content = await file.read()
    task = ingest_document.delay(file.filename, content.hex())
    return {"task_id": task.id, "status": "queued"}


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    from workers.celery_app import app as celery_app

    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "state": result.state, "result": result.result}


@app.post("/search")
def search(req: QueryRequest):
    from src.retrieval.pipeline import RetrievalPipeline

    retriever = RetrievalPipeline()
    return {"results": retriever.search(req.query, top_k=req.top_k)}


@app.post("/query")
def query(req: QueryRequest):
    from src.retrieval.rag import RAGPipeline

    rag = RAGPipeline()
    return rag.answer(req.query, top_k=req.top_k)
