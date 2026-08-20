# 진행 상황

> Kiro 세션 간 추적용. 현재 상태와 마지막 작업 기록.

---

## 현재 상태: v0.1.0 ✅ 기본 기능 완료

### 완료된 항목

| 날짜 | 항목 | 비고 |
|------|------|------|
| 2026-08-19 | 프로젝트 분석 & PROGRESS.md 생성 | 미구현 항목 식별 |
| 2026-08-20 | Dockerfile 확인 | 이미 완성 상태 |
| 2026-08-20 | 테스트 코드 43개 작성 | health, ingest, search, embedding |
| 2026-08-20 | DVC 스크립트 작성 | prepare_data.py, evaluate_retrieval.py |
| 2026-08-20 | 이미지 멀티모달 검색 | search_images, search_multimodal |
| 2026-08-20 | Celery 설정 보완 | settings 연동, timeout |
| 2026-08-20 | embedding fallback | Triton/local 자동 전환, encode_image |
| 2026-08-20 | CI 수정 | lint/format 통과, dvc-repro graceful skip |
| 2026-08-20 | docs/ 문서 체계 구축 | ROADMAP, STUDY_GUIDE, ARCHITECTURE 등 |

### 다음 작업

Phase 1 시작 (RAG 품질 개선):
1. 프롬프트 템플릿 시스템 (`src/retrieval/prompts.py`)
2. Query Rewriting 구현
3. RAGAS 평가 체계 구축

자세한 로드맵: [ROADMAP.md](./ROADMAP.md)

---

## 세션 기록

### 2026-08-20 (세션 2)
- 미구현 항목 8개 전부 구현
- CI 통과 (lint + test + docker build)
- docs/ 문서 체계 생성

### 2026-08-19 (세션 1)
- 프로젝트 구조 분석
- 미구현 항목 식별
- PROGRESS.md 초기 생성

---

## 프로젝트 요약 정보

- **API 엔드포인트**: 8개 (health, ingest, ingest/async, search, search/multimodal, search/images, query, metrics)
- **테스트**: 43개 (전부 통과)
- **CI 상태**: ✅ 전체 통과
- **모델**: BGE-M3 (1024d), SigLIP (1152d), BGE-Reranker, Qwen2-VL
- **인프라**: Milvus, MinIO, Redis, Triton, Prometheus, Grafana, ClearML
