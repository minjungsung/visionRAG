# VisionRAG 구현 진행 상황

> 이 파일은 세션 간 작업 추적용입니다. Kiro 실행 시 이 파일을 읽고 이어서 작업합니다.

## 상태 요약

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 1 | PROGRESS.md 생성 | ✅ 완료 | 세션 추적용 |
| 2 | Dockerfile (docker/Dockerfile) | ✅ 완료 | 멀티스테이지 빌드, healthcheck 포함 |
| 3 | 테스트 코드 (tests/) | ✅ 완료 | 43 tests passed (health, ingest, search, embedding) |
| 4 | DVC: scripts/prepare_data.py | ✅ 완료 | recursive char splitter, PDF 지원 |
| 5 | DVC: scripts/evaluate_retrieval.py | ✅ 완료 | MRR, P@k, R@k, NDCG@k 평가 |
| 6 | 이미지 멀티모달 검색 | ✅ 완료 | search_images, search_multimodal 추가 |
| 7 | Celery 설정 보완 | ✅ 완료 | settings 연동, acks_late, timeout 설정 |
| 8 | embedding.py fallback 연결 | ✅ 완료 | Triton↔local 자동 전환, encode_image 추가 |

## 마지막 작업 기록

- **날짜**: 2026-08-20
- **세션에서 한 일**:
  - #2 Dockerfile: 이미 완성 상태 확인 (python:3.12-slim, healthcheck, uvicorn)
  - #3 테스트: test_health, test_ingest, test_search, test_embedding (43 tests)
  - #4 prepare_data.py: txt/pdf/md 지원, recursive char splitter, JSONL 출력
  - #5 evaluate_retrieval.py: 로컬 BGE-M3로 오프라인 평가, DVC plots 호환
  - #6 이미지 검색: RetrievalPipeline에 search_images/search_multimodal 추가, API 엔드포인트 2개 추가
  - #7 Celery: settings에서 broker/backend URL 관리, autodiscover, timeout 설정
  - #8 embedding fallback: use_triton 플래그로 Triton/local 자동 전환, encode_image 메서드
- **남은 작업**: 없음 (모든 미구현 항목 완료)

## 참고: 프로젝트 구조 핵심

- API: `src/api/main.py` (FastAPI) — `/health`, `/ingest`, `/ingest/async`, `/search`, `/search/multimodal`, `/search/images`, `/query`
- 설정: `src/config/settings.py` (pydantic-settings, redis/celery URL 포함)
- 인제스천: `src/ingestion/pipeline.py`
- 검색: `src/retrieval/pipeline.py` (text + image multimodal), `src/retrieval/rag.py`
- 임베딩: `src/models/embedding.py` (Triton/local fallback, text + image)
- MLOps: `src/mlops/` (LangSmith, ClearML, Prometheus, Deepchecks)
- Workers: `workers/celery_app.py` (settings 연동), `workers/tasks.py`
- DVC: `dvc.yaml`, `scripts/prepare_data.py`, `scripts/evaluate_retrieval.py`
- CI: `.github/workflows/ci.yml` (lint-test → dvc-repro → docker-build)
- 테스트: `tests/` (pytest-asyncio, 43 tests, 모든 외부 의존성 mocked)

## 새로 추가된 API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/search/multimodal` | 텍스트+이미지 통합 검색 |
| POST | `/search/images` | 이미지 전용 검색 (SigLIP) |
