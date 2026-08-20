# VisionRAG 아키텍처

현재 시스템 구조 및 설계 의사결정 기록.

---

## 시스템 아키텍처

```
                          ┌─────────────────────────────────────────┐
                          │              Client                     │
                          └───────────────┬─────────────────────────┘
                                          │ HTTP
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI (src/api/main.py)                         │
│                                                                              │
│  /health  /ingest  /ingest/async  /search  /search/multimodal  /query        │
│                                                     /search/images           │
└────┬──────────┬────────────────────────┬─────────────────────────────────────┘
     │          │                        │
     │          ▼                        ▼
     │  ┌───────────────┐      ┌─────────────────────┐
     │  │ Celery Worker │      │  RetrievalPipeline   │
     │  │ (비동기 처리)    │      │  + RAGPipeline       │
     │  └───────┬───────┘      └──────┬──────────────┘
     │          │                     │
     │          ▼                     ▼
     │  ┌───────────────┐      ┌──────────────┐     ┌──────────────┐
     │  │IngestionPipe  │      │   Milvus     │     │   Triton     │
     │  │  (파싱+저장)    │─────▶│ (Vector DB)  │◀────│ (GPU Serve)  │
     │  └───────────────┘      └──────────────┘     └──────────────┘
     │          │                                          │
     │          ▼                                    ┌─────┴─────┐
     │  ┌───────────────┐                           │  Models    │
     │  │    MinIO      │                           │ BGE-M3     │
     │  │ (Object Store)│                           │ SigLIP     │
     │  └───────────────┘                           │ BGE-Rerank │
     │                                              │ Qwen2-VL   │
     │                                              └────────────┘
     ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Observability                            │
│  LangSmith (traces) │ Prometheus (metrics) │ Grafana (dashboard) │
│  ClearML (experiments) │ Deepchecks (validation)                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. API Layer (`src/api/main.py`)

| 엔드포인트 | 역할 |
|-----------|------|
| `GET /health` | 헬스체크 |
| `POST /ingest` | 동기 문서 인제스천 |
| `POST /ingest/async` | 비동기 인제스천 (Celery) |
| `POST /search` | 텍스트 검색 (BGE-M3 + Reranker) |
| `POST /search/multimodal` | 텍스트 + 이미지 통합 검색 |
| `POST /search/images` | 이미지 전용 검색 (SigLIP) |
| `POST /query` | RAG 답변 생성 |
| `GET /metrics` | Prometheus 메트릭 |

### 2. Ingestion Pipeline (`src/ingestion/pipeline.py`)

```
파일 업로드 → MinIO 저장 → unstructured 파싱 → 텍스트/이미지 분리
  ├─ 텍스트 → BGE-M3 임베딩 → Milvus text_chunks 저장
  └─ 이미지 → SigLIP 임베딩 → Milvus image_chunks 저장
```

### 3. Retrieval Pipeline (`src/retrieval/pipeline.py`)

```
쿼리 → BGE-M3 임베딩 → Milvus ANN 검색 (top_k*2) → BGE-Reranker 리랭킹 → top_k 반환
```

멀티모달 검색:
```
쿼리 → SigLIP 텍스트 인코더 → Milvus image_chunks 검색
     + BGE-M3 → Milvus text_chunks 검색
     → 점수 기준 병합 → top_k 반환
```

### 4. RAG Pipeline (`src/retrieval/rag.py`)

```
쿼리 → 검색 → 컨텍스트 조합 → 프롬프트 구성 → Qwen2-VL 생성 → 답변
                                                    ↓
                                            LangSmith 트레이싱
```

### 5. Embedding Model (`src/models/embedding.py`)

이중 백엔드 구조:
- `use_triton=True` → Triton gRPC로 추론 (프로덕션)
- `use_triton=False` → 로컬 SentenceTransformer (개발/테스트)
- Triton 실패 시 자동 fallback

---

## 설계 의사결정 기록 (ADR)

### ADR-001: Triton + 로컬 fallback 이중 구조
- **결정**: 임베딩을 Triton과 로컬 모델 둘 다 지원
- **이유**: GPU 없는 환경(CI, 로컬 개발)에서도 동작해야 함
- **트레이드오프**: 코드 복잡도 ↑, 하지만 DX ↑↑

### ADR-002: Milvus standalone (etcd 내장)
- **결정**: Milvus standalone 모드 사용 (클러스터 X)
- **이유**: 개인 프로젝트에서 분산 모드 불필요, 단일 docker 컨테이너로 간편
- **전환 조건**: 데이터 100M+ vectors 넘기면 클러스터 고려

### ADR-003: unstructured로 문서 파싱
- **결정**: unstructured 라이브러리 사용
- **이유**: PDF/DOCX/이미지 등 다양한 포맷 한번에 처리, 이미지 자동 추출
- **트레이드오프**: 의존성 무거움, OCR 정확도 중간

### ADR-004: DVC로 데이터 버저닝
- **결정**: DVC + MinIO remote
- **이유**: git으로 코드 관리하듯 데이터도 버전 관리, 파이프라인 재현성
- **활용**: raw → processed 변환, 평가 메트릭 추적

### ADR-005: Celery로 비동기 인제스천
- **결정**: Celery + Redis
- **이유**: 대용량 파일 인제스천은 오래 걸림, API 응답 블로킹 방지
- **대안 고려**: FastAPI BackgroundTasks (단순하지만 워커 스케일링 불가)

### ADR-006: 멀티모달 검색은 late fusion
- **결정**: 텍스트/이미지 각각 검색 후 점수 기준 병합
- **이유**: 모달리티별 독립적 최적화 가능, 구현 단순
- **대안**: early fusion (통합 임베딩) — 더 정확할 수 있지만 모델 필요

---

## 데이터 모델

### Milvus Collections

#### text_chunks
| 필드 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(64) | PK, UUID |
| doc_id | VARCHAR(64) | 원본 문서 ID |
| text | VARCHAR(8192) | 텍스트 청크 |
| embedding | FLOAT_VECTOR(1024) | BGE-M3 임베딩 |
| page_num | INT64 | 페이지 번호 |

#### image_chunks
| 필드 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(64) | PK, UUID |
| doc_id | VARCHAR(64) | 원본 문서 ID |
| image_path | VARCHAR(512) | MinIO 경로 |
| embedding | FLOAT_VECTOR(1152) | SigLIP 임베딩 |
| caption | VARCHAR(2048) | 이미지 캡션 |

### 인덱스 설정
- 타입: HNSW
- 메트릭: COSINE
- M: 16, efConstruction: 256

---

## 기술 스택 선정 이유

| 도구 | 선정 이유 | 대안 |
|------|----------|------|
| FastAPI | 비동기, 타입 힌트, OpenAPI 자동 생성 | Flask, Django |
| Milvus | 분산 벡터 DB, 필터링 지원, 성능 좋음 | Qdrant, Weaviate, pgvector |
| Triton | 멀티모델 서빙, 배치 최적화, gRPC | vLLM, TGI |
| BGE-M3 | 다국어, 1024d, 성능 우수 | E5, GTE |
| SigLIP | 이미지-텍스트 정렬, CLIP보다 성능↑ | CLIP, EVA-CLIP |
| ClearML | 오픈소스, self-hosted, 실험추적+파이프라인 | MLflow, W&B |
| DVC | git과 통합, 파이프라인 재현성 | MLflow Artifacts |
| LangSmith | RAG 특화 트레이싱, 프롬프트 관리 | Langfuse, Phoenix |
