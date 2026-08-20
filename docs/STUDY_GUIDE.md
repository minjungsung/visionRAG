# RAG / 프롬프트 엔지니어링 / 파인튜닝 학습 가이드

이 프로젝트(VisionRAG)를 실습 기반으로 활용하면서 학습하는 로드맵.

---

## Level 0: 기본 이해 (1주)

### RAG란?
```
사용자 질문 → [검색] → 관련 문서 → [LLM + 문서] → 답변
```

핵심 컴포넌트:
1. **Retriever** — 관련 문서를 찾는 것 (임베딩 유사도)
2. **Reranker** — 찾은 문서를 재정렬 (cross-encoder)
3. **Generator** — 문서를 참고해 답변 생성 (LLM)

### 우리 프로젝트에서의 위치
| 컴포넌트 | 코드 | 모델 |
|----------|------|------|
| Retriever | `src/retrieval/pipeline.py` → `_embed_query` + Milvus search | BGE-M3 |
| Reranker | `src/retrieval/pipeline.py` → `_rerank` | BGE-Reranker |
| Generator | `src/retrieval/rag.py` → `answer` | Qwen2-VL |

### 읽을 것
- [RAG 논문 원본](https://arxiv.org/abs/2005.11401) — 개념 이해용
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — 구현 패턴
- BGE-M3 논문: Multi-lingual, Multi-granularity embedding

---

## Level 1: 프롬프트 엔지니어링 (2주)

### 핵심 개념

| 기법 | 설명 | 난이도 |
|------|------|--------|
| Zero-shot | 예시 없이 지시만 | ⭐ |
| Few-shot | 좋은 예시 2-3개 포함 | ⭐⭐ |
| Chain-of-Thought | "단계별로 생각하세요" | ⭐⭐ |
| Self-Consistency | 여러 번 생성 → 다수결 | ⭐⭐⭐ |
| Tree-of-Thought | 여러 추론 경로 탐색 | ⭐⭐⭐ |

### 실습 과제

#### 과제 1: 프롬프트 템플릿 시스템 구현
```python
# src/retrieval/prompts.py 구현
class PromptTemplate:
    """질문 유형별 프롬프트 관리"""
    
    FACTUAL = """
    다음 문서를 참고하여 사실에 기반해 답하세요.
    반드시 [출처 n] 형식으로 근거를 표시하세요.
    문서에 없는 내용은 "확인할 수 없습니다"라고 답하세요.
    
    [문서]
    {context}
    
    [질문] {query}
    [답변]
    """
    
    ANALYTICAL = """
    다음 문서를 분석하여 질문에 답하세요.
    1. 먼저 관련 정보를 정리하세요.
    2. 여러 관점에서 분석하세요.
    3. 결론을 도출하세요.
    
    [문서]
    {context}
    
    [질문] {query}
    [분석]
    """
```

#### 과제 2: 프롬프트 변형별 성능 비교
- LangSmith에서 동일 질문 세트로 프롬프트 A vs B 비교
- Faithfulness (환각 여부), Relevancy (답변 관련성) 측정

#### 과제 3: Citation 강제
- 답변에 `[1]`, `[2]` 등 출처 번호 붙이기
- 출처 없는 주장이 있으면 패널티

### 읽을 것
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- DSPY 논문 — 프롬프트 자동 최적화

---

## Level 2: Advanced RAG (3-4주)

### 핵심 기법들

#### Query Rewriting
```
원본: "작년에 매출 어땠어?"
→ 변환: "2025년 연간 매출 실적 및 전년 대비 성장률"
```
- LLM으로 검색 최적화된 쿼리 생성
- 모호한 대명사 해소, 구체화

#### HyDE (Hypothetical Document Embeddings)
```
질문 → LLM이 가상 답변 생성 → 가상 답변으로 검색
```
- 질문과 문서의 임베딩 공간 gap을 줄임
- 특히 짧은 질문에 효과적

#### Self-RAG
```
질문 → 검색 필요한지 판단 → 검색 → 결과 충분한지 판단 → 부족하면 재검색
```
- LLM이 스스로 retrieval 필요성 판단
- 불필요한 검색 줄이고, 필요할 때만 검색

#### Chunking 전략 비교

| 전략 | 장점 | 단점 |
|------|------|------|
| Fixed size (현재) | 단순, 빠름 | 의미 단위 안 맞을 수 있음 |
| Semantic | 의미 보존 | 느림, 청크 크기 불균일 |
| Parent-Child | 정확한 검색 + 넓은 컨텍스트 | 구현 복잡 |
| Sentence window | 문장 단위 검색 + 주변 문장 포함 | 메모리 |

### 실습 과제

#### 과제 4: Query Rewriting 구현
- `src/retrieval/query_rewriter.py` 추가
- 원본 쿼리 → LLM으로 검색용 쿼리 변환
- evaluate_retrieval.py에서 원본 vs rewritten 비교

#### 과제 5: HyDE 구현
- 질문 → Qwen2-VL로 가상 답변 생성 → 가상 답변 임베딩으로 검색
- 짧은 질문 / 긴 질문 각각 효과 비교

#### 과제 6: Chunking 전략 실험
- `scripts/prepare_data.py`에 semantic chunking 옵션 추가
- DVC로 전략별 메트릭 비교

### RAGAS 평가 프레임워크

```python
# 평가 메트릭
- Faithfulness: 답변이 검색된 문서에 근거하는가?
- Answer Relevancy: 답변이 질문에 맞는가?
- Context Precision: 검색된 문서가 관련 있는가?
- Context Recall: 필요한 정보를 다 찾았는가?
```

실습: `scripts/evaluate_rag.py` 구현 (RAGAS + LangSmith 연동)

### 읽을 것
- [RAGAS 문서](https://docs.ragas.io/)
- [Advanced RAG Techniques (LlamaIndex)](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- HyDE 논문: Precise Zero-Shot Dense Retrieval without Relevance Labels
- Self-RAG 논문: Self-Reflective Retrieval-Augmented Generation

---

## Level 3: 임베딩 파인튜닝 (4-6주)

### 왜 파인튜닝?
- 프리트레인 모델은 일반적인 유사도만 알음
- 도메인 용어, 약어, 특수 관계는 학습 안 됨
- 예: "K8s" ≈ "Kubernetes" 연결이 약할 수 있음

### 학습 데이터 구축

#### Synthetic Data Generation
```python
# LLM으로 QA 페어 자동 생성
document = "Milvus는 분산 벡터 데이터베이스로..."
prompt = f"""
이 문서에 대해 질문-답변 쌍 5개를 생성하세요.
다양한 난이도와 유형으로 만드세요.
[문서] {document}
"""
```

#### Hard Negative Mining
```python
# 비슷하지만 관련 없는 문서 찾기
query_embedding = model.encode(query)
all_results = milvus.search(query_embedding, limit=100)
hard_negatives = [r for r in all_results[10:30] if r.doc_id not in relevant_ids]
```

### Contrastive Learning
```
Loss = -log(sim(query, positive) / (sim(query, positive) + Σ sim(query, negative)))
```
- (query, positive_doc) 페어를 가깝게
- (query, negative_doc) 페어를 멀게

### 실습 과제

#### 과제 7: 학습 데이터 생성
- `scripts/generate_training_data.py`
- 기존 문서에서 LLM으로 QA 페어 생성
- Hard negative mining으로 삼중쌍 (query, pos, neg) 구축

#### 과제 8: BGE-M3 파인튜닝
- `scripts/train_embeddings.py`
- sentence-transformers의 MultipleNegativesRankingLoss
- ClearML로 loss curve, eval metrics 추적

#### 과제 9: 성능 비교
- 파인튜닝 전 vs 후 MRR, NDCG 비교 (DVC metrics)
- 도메인 질문 vs 일반 질문 각각 측정

### 읽을 것
- [Sentence-BERT Fine-tuning Guide](https://www.sbert.net/docs/sentence_transformer/training_overview.html)
- BGE 논문: C-Pack (Contrastive Pre-training for Embedding)
- [FlagEmbedding 학습 코드](https://github.com/FlagOpen/FlagEmbedding)

---

## Level 4: 리랭커 & LLM 파인튜닝 (6-8주)

### 리랭커 파인튜닝
- Cross-encoder 구조로 (query, passage) 쌍의 관련도 직접 예측
- 검색 로그 기반 preference data 구축
- Loss: Binary Cross Entropy or Margin Ranking Loss

### LLM 답변 품질 개선
- **SFT (Supervised Fine-Tuning)**: 좋은 답변 예시로 학습
- **DPO (Direct Preference Optimization)**: 좋은 답변 vs 나쁜 답변 페어로 학습
- **RLHF**: Human feedback 기반 (비용 높음)

### 실습 과제

#### 과제 10: 리랭커 파인튜닝
- 검색 로그에서 (query, clicked_doc, skipped_doc) 삼중쌍 생성
- BGE-Reranker fine-tune

#### 과제 11: LLM DPO
- 동일 질문에 대한 좋은/나쁜 답변 쌍 생성
- Qwen2-VL LoRA fine-tune + DPO

### 읽을 것
- DPO 논문: Direct Preference Optimization
- LoRA 논문: Low-Rank Adaptation of Large Language Models
- QLoRA: Efficient Fine-tuning of Quantized LLMs

---

## 학습 원칙

1. **측정 먼저** — 개선하려면 현재 성능을 먼저 알아야 함 (RAGAS, MRR)
2. **한 번에 하나씩** — 변수 하나만 바꾸고 비교
3. **기록 필수** — ClearML/LangSmith/docs/EXPERIMENTS.md에 실험 기록
4. **실패도 기록** — 안 된 것도 왜 안 됐는지 적기
5. **이 프로젝트에서 실습** — 이론만 읽지 말고 코드로 구현

---

## 추천 순서 (이 프로젝트 기준)

```
1. 프롬프트 템플릿 시스템 구현 (Level 1, 과제 1-3)
   → 가장 빠르게 효과 볼 수 있음
   
2. RAGAS 평가 체계 구축 (Level 2)
   → 이후 모든 실험의 기준선 확보

3. Query Rewriting + HyDE (Level 2, 과제 4-5)
   → 검색 품질 개선, 코드량 적음

4. 임베딩 파인튜닝 (Level 3, 과제 7-9)
   → 가장 근본적인 개선, 시간 필요

5. 리랭커/LLM 파인튜닝 (Level 4)
   → 임베딩이 좋아진 후에 의미 있음
```
