# VisionRAG

멀티모달 지식 관리 시스템 — 텍스트 + 이미지 통합 검색 및 AI 답변 생성

## 아키텍처

```
┌─────────────┐    ┌────────── ┐    ┌──────────── ┐
│  FastAPI    │───▶│  Milvus   │    │   Triton    │
│  + Metrics  │    │(Vector DB)│    │(GPU Serving)│
└──────┬──────┘    └────────── ┘    └──────────── ┘
       │
       │  traces           experiments        data versions
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  LangSmith   │   │   ClearML    │   │  DVC + MinIO │
│  (Observ.)   │   │  (Tracking)  │   │ (Versioning) │
└──────────────┘   └──────────────┘   └──────────────┘
       │
       ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Prometheus  │──▶│   Grafana    │   │  Deepchecks  │
│  (Metrics)   │   │ (Dashboard)  │   │ (Validation) │
└──────────────┘   └──────────────┘   └──────────────┘
```

## 스택

| 영역 | 도구 |
|------|------|
| Backend | FastAPI |
| Vector DB | Milvus |
| Model Serving | NVIDIA Triton Inference Server |
| Storage | MinIO (S3 호환) |
| Task Queue | Celery + Redis |
| Models | BGE-M3, SigLIP, BGE-Reranker, Qwen2-VL |
| LLM Observability | LangSmith |
| Experiment Tracking | ClearML |
| Data Versioning | DVC (MinIO remote) |
| Monitoring | Prometheus + Grafana |
| Data Validation | Deepchecks |
| CI/CD | GitHub Actions |

## 실행

```bash
# 1. 인프라 (DB, Storage, Queue)
docker compose up -d milvus minio redis

# 2. Milvus 컬렉션 초기화 (최초 1회)
source .venv/bin/activate
PYTHONPATH=. python scripts/init_milvus.py

# 3. API 서버
uvicorn src.api.main:app --reload --port 8080

# 4. Web UI (Gradio)
PYTHONPATH=. python app.py
# 브라우저에서 http://localhost:7860

# 5. 헬스체크
curl http://localhost:8080/health
```

### GPU 없이 로컬 모드 (기본)

`.env`에 `USE_TRITON=false` (기본값)이면 로컬 sentence-transformers 모델 사용.  
Docker로 Milvus + MinIO + Redis만 띄우면 전체 파이프라인 동작:

```bash
docker compose up -d milvus minio redis
PYTHONPATH=. python scripts/init_milvus.py
uvicorn src.api.main:app --reload --port 8080
```

### 전체 인프라 (GPU 있을 때)

```bash
docker compose up -d  # 전부 띄움 (Triton 포함)
# .env에 USE_TRITON=true 설정
```

## MLOps 설정

### LangSmith

`.env`에 추가:
```env
LANGSMITH_API_KEY=your-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=visionrag
```

### ClearML

```bash
docker compose up -d clearml-server
# Web UI: http://localhost:8008
```

### DVC

```bash
pip install -e ".[mlops]"
dvc pull              # 데이터 다운로드
dvc repro             # 파이프라인 실행
dvc push              # 결과물 업로드
```

### Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- FastAPI Metrics: http://localhost:8080/metrics

### Data Validation

```python
from src.mlops.validation import validate_embeddings, check_data_drift

result = validate_embeddings(embeddings)
drift = check_data_drift(ref_embeddings, cur_embeddings)
```

## 개발 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlops]"
```

## 프로젝트 구조

```
src/
├── api/            # FastAPI 엔드포인트
├── config/         # 설정 (pydantic-settings)
├── ingestion/      # 문서 수집 파이프라인
├── models/         # Triton model repository
├── mlops/          # MLOps 통합 코드
│   ├── __init__.py          # LangSmith tracing
│   ├── clearml_tracking.py  # 실험 추적
│   ├── metrics.py           # Prometheus 메트릭
│   └── validation.py        # Deepchecks 검증
└── retrieval/      # 검색 + RAG 파이프라인
monitoring/
├── prometheus.yml
└── grafana/
    ├── dashboards/
    └── provisioning/
.dvc/               # DVC 설정
.github/workflows/  # CI/CD
```

## 문서

| 문서 | 내용 |
|------|------|
| [docs/USAGE.md](docs/USAGE.md) | **사용 가이드 (시작하기, UI, API, 평가 등)** |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 프로젝트 TODO 및 기능 개발 로드맵 |
| [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) | RAG/프롬프트엔지니어링/파인튜닝 학습 가이드 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 시스템 아키텍처 및 설계 의사결정 |
| [docs/COMPONENTS.md](docs/COMPONENTS.md) | 컴포넌트별 역할 요약 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 진행 상황 추적 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 실험 기록 |
| [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | 다음 구현 계획 (프롬프트 템플릿 + Query Rewriting + RAGAS) |
