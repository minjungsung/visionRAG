# VisionRAG

멀티모달 지식 관리 시스템 — 텍스트 + 이미지 통합 검색 및 AI 답변 생성

## 스택

- **Backend**: FastAPI
- **Vector DB**: Milvus
- **Model Serving**: NVIDIA Triton Inference Server
- **Storage**: MinIO (S3 호환)
- **Task Queue**: Celery + Redis
- **Models**: BGE-M3 (텍스트 임베딩), SigLIP (이미지 임베딩), BGE-Reranker, Qwen2-VL

## 실행

```bash
# 1. 인프라
docker compose up -d milvus minio redis

# 2. Triton (GPU 필요)
docker compose up -d triton

# 3. API 서버
uvicorn src.api.main:app --reload --port 8080

# 4. 헬스체크
curl http://localhost:8080/health
```

## 개발 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
