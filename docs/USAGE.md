# VisionRAG 사용 가이드

---

## 사전 준비

```bash
# 1. 가상환경 + 패키지 설치
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlops]"

# 2. 인프라 띄우기 (Milvus, MinIO, Redis)
docker compose up -d milvus minio redis

# 3. Milvus 컬렉션 초기화 (최초 1회)
python scripts/init_milvus.py

# 4. .env 설정 (BCAI API 사용 시)
cat .env
# OPENAI_API_KEY=your-bcai-pat
# OPENAI_MODEL=gpt-5.1-codex-mini
# OPENAI_BASE_URL=https://bcai-openai-proxy-test.taspre-phx.apps.boeing.com/v1
```

---

## 1. 웹 UI (질문 + 검색)

```bash
python app.py
```

브라우저에서 http://localhost:7860 접속.

| 탭 | 기능 |
|----|------|
| 💬 RAG 질의 | 질문 입력 → AI 답변 + 출처 문서 |
| 🔎 검색 | 벡터 검색만 (답변 없이, 빠름) |
| ℹ️ 시스템 정보 | 현재 설정 확인 + 사용법 |

**프롬프트 타입 옵션:**
- `default` — 일반 질의
- `factual` — 사실 확인 ("~는 뭐야?")
- `analytical` — 분석 ("왜 ~인가?")
- `comparative` — 비교 ("A vs B?")
- `how_to` — 방법 ("~하려면?")

---

## 2. API 서버

```bash
uvicorn src.api.main:app --reload --port 8080
```

### 엔드포인트

```bash
# 헬스체크
curl http://localhost:8080/health

# RAG 질의 (답변 생성)
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "이미지 검색 어떻게 동작해?", "top_k": 5, "prompt_type": "how_to"}'

# 텍스트 검색 (답변 없이)
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Milvus HNSW 인덱스", "top_k": 10}'

# 멀티모달 검색 (텍스트 + 이미지 통합)
curl -X POST http://localhost:8080/search/multimodal \
  -H "Content-Type: application/json" \
  -d '{"query": "시스템 아키텍처 다이어그램", "top_k": 5}'

# 이미지 전용 검색
curl -X POST http://localhost:8080/search/images \
  -H "Content-Type: application/json" \
  -d '{"query": "파이프라인 흐름도", "top_k": 5}'

# 문서 인제스천 (동기)
curl -X POST http://localhost:8080/ingest \
  -F "file=@document.pdf"

# 문서 인제스천 (비동기)
curl -X POST http://localhost:8080/ingest/async \
  -F "file=@large_document.pdf"

# 비동기 작업 상태 확인
curl http://localhost:8080/tasks/{task_id}

# Prometheus 메트릭
curl http://localhost:8080/metrics
```

---

## 3. 문서 인제스천

### 웹 UI 없이 직접 넣기

```bash
# 테스트 데이터 (golden_qa.jsonl 기반)
python scripts/ingest_test_data.py

# API로 파일 업로드
curl -X POST http://localhost:8080/ingest -F "file=@my_document.pdf"
```

### 지원 포맷
- PDF, DOCX, TXT, Markdown
- 이미지 (PNG, JPG) — SigLIP로 임베딩

---

## 4. 품질 평가

### 검색 품질 비교 (Rewrite 전략)

```bash
python scripts/experiment_rewrite.py
```

- `none` vs `simple` (규칙 기반 약어 전개) 비교
- Hit Rate, Top-1 Score, Latency 측정
- 결과: `reports/experiment_rewrite_comparison.json`

### End-to-End RAG 평가 (LLM-as-judge)

```bash
python scripts/evaluate_rag_simple.py
```

- 20개 QA로 전체 파이프라인 평가
- 메트릭: Faithfulness, Answer Relevancy, Context Relevancy
- BCAI API를 judge로 사용
- 결과: `reports/rag_metrics.json`, `reports/rag_per_query.json`

### Retrieval 메트릭 (DVC)

```bash
dvc repro evaluate_retrieval
```

- MRR, Precision@K, Recall@K, NDCG@K
- 결과: `reports/retrieval_metrics.json`

---

## 5. Query Rewriting 설정

`.env`에서 전략 변경:

```env
# 끄기 (기본값)
REWRITE_STRATEGY=none

# 규칙 기반 (약어 전개, 불용어 제거)
REWRITE_STRATEGY=simple

# 여러 변형으로 확장 검색
REWRITE_STRATEGY=multi_query

# 가상 답변 생성 후 검색 (LLM 필요)
REWRITE_STRATEGY=hyde
```

LLM 기반 rewrite 사용 시:
```env
REWRITE_OPENAI_API_KEY=your-key
REWRITE_LLM_MODEL=gpt-5.1-codex-mini
```

---

## 6. 모니터링

### LangSmith (RAG 트레이싱)

`.env` 설정:
```env
LANGSMITH_API_KEY=lsv2_pt_xxxxx
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=visionrag
```

https://smith.langchain.com 에서 프로젝트 `visionrag` 확인.

### Prometheus + Grafana

```bash
docker compose up -d prometheus grafana
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- API 메트릭: http://localhost:8080/metrics

### ClearML (실험 추적)

```bash
docker compose up -d clearml-server
```

- Web UI: http://localhost:8008

---

## 7. 개발 / 테스트

```bash
# 테스트 실행 (87개)
pytest

# lint
ruff check .

# 포맷팅
ruff format .
```

---

## 8. DVC 파이프라인

```bash
# 전체 파이프라인 실행
dvc repro

# 데이터 다운로드
dvc pull

# 결과물 업로드
dvc push

# 메트릭 비교
dvc metrics show
dvc metrics diff
```

파이프라인 스테이지:
```
prepare_data → evaluate_retrieval → evaluate_rag
```

---

## 파일 구조 요약

```
app.py                          ← 웹 UI (Gradio)
src/api/main.py                 ← REST API (FastAPI)
src/retrieval/rag.py            ← RAG 답변 생성
src/retrieval/pipeline.py       ← 벡터 검색 + 리랭킹
src/retrieval/prompts.py        ← 프롬프트 템플릿
src/retrieval/query_rewriter.py ← 쿼리 리라이팅
src/models/embedding.py         ← 임베딩 모델
src/config/settings.py          ← 설정 (.env 로드)
scripts/
├── init_milvus.py              ← 컬렉션 초기화
├── ingest_test_data.py         ← 테스트 데이터 인제스천
├── experiment_rewrite.py       ← 검색 실험
├── evaluate_rag_simple.py      ← RAG 평가 (LLM-as-judge)
├── evaluate_rag.py             ← RAG 평가 (RAGAS, OpenAI 필요)
├── evaluate_retrieval.py       ← 검색 메트릭
└── prepare_data.py             ← 데이터 전처리
```

---

## 빠른 시작 (처음부터)

```bash
# 1. 설치
pip install -e ".[dev]"

# 2. 인프라
docker compose up -d milvus minio redis

# 3. 초기화
python scripts/init_milvus.py
python scripts/ingest_test_data.py

# 4. UI 실행
python app.py
# → http://localhost:7860 에서 질문하기
```
