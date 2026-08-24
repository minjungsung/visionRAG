# VisionRAG 사용 가이드

로컬에서 전체 시스템을 띄우고 문서 업로드 → 검색 → AI 답변 받는 방법.

---

## 빠른 시작 (5분)

### 1. 인프라 띄우기

```bash
# Docker 필요 (Docker Desktop 켜져 있어야 함)
cd /Users/ah236f/Projects/Personal/visionrag

# 핵심 인프라: Milvus(벡터DB) + MinIO(파일저장) + Redis(큐)
docker compose up -d milvus minio redis
```

확인:
```bash
# Milvus 정상?
curl http://localhost:9091/healthz
# MinIO 콘솔 (브라우저에서 열기)
open http://localhost:9002   # ID: minioadmin / PW: minioadmin
```

### 2. Milvus 컬렉션 생성

```bash
source .venv/bin/activate
python scripts/init_milvus.py
```
출력: `Created collection: text_chunks`, `Created collection: image_chunks`

### 3. API 서버 실행

```bash
uvicorn src.api.main:app --reload --port 8080
```

### 4. 헬스체크

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

---

## 문서 업로드 (인제스천)

### 텍스트/PDF 파일 업로드

```bash
# 동기 (완료될 때까지 기다림)
curl -X POST http://localhost:8080/ingest \
  -F "file=@/path/to/document.pdf"

# 응답
# {"doc_id": "550e8400-e29b-41d4-a716-446655440000"}
```

```bash
# 비동기 (큐에 넣고 바로 리턴) — Redis + Celery 워커 필요
curl -X POST http://localhost:8080/ingest/async \
  -F "file=@/path/to/large_document.pdf"

# 응답
# {"task_id": "abc-123", "status": "queued"}

# 상태 확인
curl http://localhost:8080/tasks/abc-123
# {"task_id": "abc-123", "state": "SUCCESS", "result": {"doc_id": "..."}}
```

### 지원 파일 형식
- PDF (.pdf)
- 텍스트 (.txt)
- 마크다운 (.md)
- Word (.docx) — unstructured 설치 시

---

## 검색

### 텍스트 검색

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "벡터 데이터베이스 사용법", "top_k": 5}'
```

응답:
```json
{
  "results": [
    {
      "text": "Milvus는 오픈소스 벡터 데이터베이스로...",
      "score": 0.92,
      "doc_id": "550e8400-...",
      "page_num": 3,
      "type": "text"
    },
    ...
  ]
}
```

### 이미지 검색

```bash
curl -X POST http://localhost:8080/search/images \
  -H "Content-Type: application/json" \
  -d '{"query": "시스템 아키텍처 다이어그램", "top_k": 3}'
```

### 멀티모달 검색 (텍스트 + 이미지 통합)

```bash
curl -X POST http://localhost:8080/search/multimodal \
  -H "Content-Type: application/json" \
  -d '{"query": "RAG 파이프라인 구조", "top_k": 5}'
```

---

## AI 답변 (RAG Query)

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "VisionRAG에서 이미지 검색은 어떻게 동작해?", "top_k": 5}'
```

응답:
```json
{
  "answer": "VisionRAG의 이미지 검색은 SigLIP 모델을 사용하여...",
  "sources": [
    {"text": "...", "score": 0.95, "doc_id": "...", "page_num": 1}
  ]
}
```

---

## 모니터링

### Prometheus 메트릭

```bash
curl http://localhost:8080/metrics
```

### Grafana 대시보드

```bash
docker compose up -d prometheus grafana
open http://localhost:3000   # ID: admin / PW: admin
```

### LangSmith 트레이싱

`.env`에 이미 설정됨. `/query` 호출할 때마다 자동으로 트레이스 전송.

확인: https://smith.langchain.com → visionrag 프로젝트

---

## 전체 인프라 띄우기 (GPU 있을 때)

```bash
# 1. 기본 인프라
docker compose up -d milvus minio redis

# 2. MLOps
docker compose up -d clearml-server prometheus grafana

# 3. Triton (GPU 필요 — NVIDIA Docker)
docker compose up -d triton

# 4. Milvus 초기화
python scripts/init_milvus.py

# 5. API 서버
uvicorn src.api.main:app --reload --port 8080

# 6. Celery 워커 (비동기 인제스천용)
celery -A workers.celery_app worker --loglevel=info
```

---

## GPU 없이 쓰기 (로컬 개발 모드)

`.env`에 `USE_TRITON=false` (기본값)이면 로컬 모델 사용:
- BGE-M3 → sentence-transformers로 로컬 실행 (첫 로딩 시 다운로드)
- SigLIP → open_clip 또는 sentence-transformers

⚠️ 단, 현재 `RetrievalPipeline`과 `IngestionPipeline`은 직접 Triton 클라이언트를 사용하므로,
**로컬 전용 파이프라인**을 쓰려면 `EmbeddingModel`을 직접 호출하는 방식으로 코드 수정 필요.
(이건 Phase 1에서 리팩토링 예정 — [ROADMAP.md](./ROADMAP.md) 참고)

현재 로컬에서 할 수 있는 것:
```bash
# 테스트 실행
pytest tests/ -v

# DVC 파이프라인 (검색 평가 — 로컬 모델)
python scripts/prepare_data.py
python scripts/evaluate_retrieval.py

# API 서버 (health, metrics만 동작)
uvicorn src.api.main:app --reload --port 8080
curl http://localhost:8080/health
```

---

## Python에서 직접 사용

```python
# 임베딩 모델 (로컬, GPU/Triton 불필요)
from src.models.embedding import embedding_model

texts = ["안녕하세요", "벡터 검색 테스트"]
embeddings = embedding_model.encode(texts)
print(embeddings.shape)  # (2, 1024)

# DVC 데이터 평가
from scripts.evaluate_retrieval import compute_embeddings, retrieve_top_k
```

---

## API 문서 (Swagger)

서버 실행 후:
```
http://localhost:8080/docs     # Swagger UI
http://localhost:8080/redoc    # ReDoc
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Connection refused :19530` | Milvus 안 띄움 | `docker compose up -d milvus` |
| `Triton not live` | Triton 안 띄움 or GPU 없음 | `.env`에 `USE_TRITON=false` |
| `No module 'unstructured'` | OCR 의존성 없음 | `pip install -e ".[ocr]"` |
| `LANGSMITH_API_KEY empty` | .env 설정 안 됨 | `.env` 확인 |
| 인제스천 느림 | 동기 모드 | `/ingest/async` + Celery 워커 사용 |
| 첫 검색 느림 | 모델 다운로드 중 | 최초 1회만, 이후 캐시됨 |

---

## 요약: 최소 실행 조합

| 할 일 | 필요한 것 |
|--------|----------|
| 테스트 돌리기 | venv만 있으면 됨 |
| 검색 평가 | venv + 모델 다운로드 (자동) |
| 문서 업로드 + 검색 | Docker (Milvus + MinIO) + Triton or 로컬 모델 |
| AI 답변 | 위 + Qwen2-VL (Triton) |
| 모니터링 | 위 + Prometheus + Grafana |
| 비동기 인제스천 | 위 + Redis + Celery 워커 |
