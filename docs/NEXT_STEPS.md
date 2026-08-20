# 다음 구현: 프롬프트 템플릿 + Query Rewriting + RAGAS 평가

Phase 1에서 바로 시작할 3가지 기능의 구현 계획.

---

## 왜 이 조합인가?

1. **프롬프트 템플릿** → 프롬프트 변형 실험을 쉽게 할 수 있는 기반
2. **Query Rewriting** → 검색 품질 즉시 개선 (코드량 적음, 효과 큼)
3. **RAGAS 평가** → 개선 효과를 정량적으로 증명 (측정 없이 개선 없음)

보너스: LangSmith 트레이싱이 이미 있으니 디버깅도 편함.

---

## 1. 프롬프트 템플릿 시스템

### 파일: `src/retrieval/prompts.py`

```python
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    FACTUAL = "factual"       # 사실 확인형
    ANALYTICAL = "analytical"  # 분석형
    COMPARATIVE = "comparative"  # 비교형
    HOW_TO = "how_to"         # 방법 질문


@dataclass
class PromptConfig:
    system_prompt: str
    user_template: str
    require_citation: bool = True
    chain_of_thought: bool = False
    few_shot_examples: list[dict] | None = None
```

### 핵심 기능:
- 질문 유형별 프롬프트 자동 선택
- Citation 강제 옵션
- CoT 활성화 옵션
- Few-shot 예시 관리
- LangSmith에서 버전별 추적

### 수정 대상:
- `src/retrieval/rag.py` — PromptConfig 사용하도록 변경
- `src/api/main.py` — `/query` 엔드포인트에 prompt_type 파라미터 추가

---

## 2. Query Rewriting

### 파일: `src/retrieval/query_rewriter.py`

```python
class QueryRewriter:
    """검색 최적화를 위한 쿼리 리라이팅."""
    
    def rewrite(self, query: str) -> str:
        """LLM으로 검색에 최적화된 쿼리 변환."""
        ...
    
    def expand(self, query: str, n: int = 3) -> list[str]:
        """쿼리를 여러 변형으로 확장 (multi-query)."""
        ...
    
    def hyde(self, query: str) -> str:
        """HyDE: 가상 답변 생성 후 그걸로 검색."""
        ...
```

### 전략:
1. **Simple Rewrite** — 모호한 표현 구체화, 약어 전개
2. **Multi-Query** — 하나의 질문을 여러 관점으로 확장 → 각각 검색 → union
3. **HyDE** — LLM이 가상 답변 생성 → 가상 답변의 임베딩으로 검색

### 수정 대상:
- `src/retrieval/pipeline.py` — search() 시작에 rewrite 단계 추가 (옵션)
- `src/config/settings.py` — rewrite 관련 설정 추가

---

## 3. RAGAS 평가

### 파일: `scripts/evaluate_rag.py`

```python
# RAGAS 메트릭:
# - Faithfulness: 답변이 검색된 문서에 근거하는가? (환각 체크)
# - Answer Relevancy: 답변이 질문에 적절한가?
# - Context Precision: 검색된 문서가 실제로 관련 있는가?
# - Context Recall: 필요한 정보를 빠짐없이 찾았는가?
```

### 필요 사항:
- `data/golden_qa.jsonl` — 정답이 있는 QA 페어 (수동 구축 or LLM 생성)
- `ragas` 패키지 추가 (pyproject.toml)
- LangSmith 연동으로 평가 결과 시각화

### 형식:
```jsonl
{"query": "VisionRAG에서 이미지 검색은 어떻게 동작하나요?", "ground_truth": "SigLIP 모델로...", "relevant_contexts": ["..."]}
```

### DVC 파이프라인 추가:
```yaml
# dvc.yaml에 추가
evaluate_rag:
  cmd: python scripts/evaluate_rag.py
  deps:
    - scripts/evaluate_rag.py
    - data/golden_qa.jsonl
  metrics:
    - reports/rag_metrics.json:
        cache: false
```

---

## 구현 순서

```
Step 1: 프롬프트 템플릿 시스템
  └─ src/retrieval/prompts.py 생성
  └─ src/retrieval/rag.py 수정
  └─ 테스트 추가

Step 2: RAGAS 평가 (측정 기반 먼저 갖추기)
  └─ ragas 의존성 추가
  └─ scripts/evaluate_rag.py 생성
  └─ data/golden_qa.jsonl 초기 구축 (20개)
  └─ Baseline 측정

Step 3: Query Rewriting
  └─ src/retrieval/query_rewriter.py 생성
  └─ pipeline.py에 옵션으로 통합
  └─ RAGAS로 개선 효과 측정

Step 4: 실험 & 비교
  └─ 프롬프트 A/B 비교 (EXP-001)
  └─ Rewrite vs No-Rewrite 비교 (EXP-002)
  └─ EXPERIMENTS.md 기록
```

---

## 의존성 추가 필요

```toml
# pyproject.toml [project.optional-dependencies]
eval = [
    "ragas>=0.1",
    "datasets>=2.0",
]
```

---

## 시작하려면

Kiro에게: "Phase 1 Step 1부터 구현 시작해줘" 라고 하면 바로 작업 시작.
