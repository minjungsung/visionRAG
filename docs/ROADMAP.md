# VisionRAG 로드맵

프로젝트 기능 개발 우선순위 및 TODO 관리.

---

## 현재 상태 (v0.1.0)

✅ 완료된 기능:
- FastAPI 서버 (health, ingest, search, query)
- 멀티모달 검색 (텍스트 + 이미지)
- Milvus 벡터 DB 연동
- Triton Inference Server 연동 (BGE-M3, SigLIP, BGE-Reranker, Qwen2-VL)
- 로컬 임베딩 fallback (Triton 없을 때)
- Celery 비동기 인제스천
- DVC 파이프라인 (prepare_data, evaluate_retrieval)
- MLOps 기반 (LangSmith, ClearML, Prometheus, Grafana, Deepchecks)
- CI/CD (lint + test + docker build)
- 테스트 43개 (전부 통과)

---

## Phase 1: RAG 품질 개선 (다음 목표)

### 1.1 프롬프트 엔지니어링 시스템
- [ ] 프롬프트 템플릿 매니저 (`src/retrieval/prompts.py`)
- [ ] 질문 유형 분류기 (factual / analytical / creative)
- [ ] Chain-of-Thought 프롬프트 옵션
- [ ] Few-shot 예시 관리 시스템
- [ ] Citation 강제 프롬프트 (출처 표시)
- [ ] LangSmith에서 프롬프트 버전별 성능 비교

### 1.2 Advanced RAG 기법
- [ ] Query Rewriting — 쿼리를 검색에 최적화된 형태로 변환
- [ ] HyDE (Hypothetical Document Embeddings) — 가상 답변으로 검색
- [ ] Multi-step RAG — 1차 검색 → 부족하면 재검색
- [ ] Parent-Child Chunking — 작은 청크로 검색, 큰 청크로 컨텍스트 제공
- [ ] Semantic Chunking — 의미 단위 분리 (고정 길이 X)

### 1.3 평가 체계
- [ ] RAGAS 연동 (faithfulness, relevancy, context precision)
- [ ] 자동 평가 파이프라인 (`scripts/evaluate_rag.py`)
- [ ] Golden dataset 구축 (100+ QA 페어)
- [ ] A/B 테스트 프레임워크

---

## Phase 2: 모델 파인튜닝

### 2.1 임베딩 모델 파인튜닝
- [ ] 학습 데이터 생성 (LLM으로 synthetic QA 페어)
- [ ] Hard negative mining 파이프라인
- [ ] BGE-M3 도메인 fine-tune 스크립트
- [ ] 파인튜닝 전/후 성능 비교 (DVC metrics)
- [ ] ClearML 실험 추적 연동

### 2.2 리랭커 파인튜닝
- [ ] 검색 로그 → 학습 데이터 변환
- [ ] Cross-encoder fine-tune
- [ ] LLM-as-judge vs Cross-encoder 성능 비교

### 2.3 모델 배포
- [ ] ONNX 변환 스크립트 (`scripts/export_model.py`)
- [ ] Triton model repository 자동 업데이트
- [ ] 모델 버저닝 (DVC로 weight 관리)

---

## Phase 3: 프로덕션 고도화

### 3.1 피드백 루프
- [ ] 사용자 피드백 수집 API (`POST /feedback`)
- [ ] 피드백 → 학습 데이터 변환 파이프라인
- [ ] Active Learning 기반 데이터 선별

### 3.2 성능 최적화
- [ ] 임베딩 캐시 (Redis)
- [ ] 결과 캐싱 (자주 묻는 질문)
- [ ] 배치 인제스천 최적화
- [ ] 모델 Quantization (INT8)

### 3.3 데이터 파이프라인
- [ ] 자동 크롤링 스케줄러 (Celery beat)
- [ ] OCR 고도화 (테이블/차트 추출)
- [ ] 문서 중복 감지 + 자동 업데이트

### 3.4 모니터링 & 안정성
- [ ] Data drift 자동 감지 (Deepchecks 스케줄링)
- [ ] 알럿 설정 (Grafana → Slack)
- [ ] 모델 성능 degradation 자동 감지
- [ ] 롤백 자동화

---

## Phase 4: 확장

- [ ] Multi-tenant 지원
- [ ] Graph RAG (문서 간 관계 그래프)
- [ ] Streaming 답변 (SSE)
- [ ] 대화형 RAG (멀티턴 컨텍스트)
- [ ] Agent 기반 RAG (tool use)

---

## 우선순위 결정 기준

1. **검색 품질에 직접 영향** → Phase 1, 2
2. **정량 측정 가능** → 평가 체계 먼저
3. **기존 인프라 활용** → ClearML, LangSmith, DVC 이미 있으니 활용

> 💡 추천 시작점: Phase 1.1 (프롬프트 템플릿) + Phase 1.3 (RAGAS 평가) 먼저 → 측정할 수 있어야 개선할 수 있다.
