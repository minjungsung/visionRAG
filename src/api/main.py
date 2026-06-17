from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.config.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: DB 연결 등
    yield
    # shutdown: 정리

app = FastAPI(title="VisionRAG", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}