"""검색 최적화를 위한 쿼리 리라이팅.

3가지 전략을 제공합니다:
1. Simple Rewrite — 모호한 표현 구체화, 약어 전개
2. Multi-Query — 하나의 질문을 여러 관점으로 확장 → 각각 검색 → union
3. HyDE — LLM이 가상 답변 생성 → 가상 답변의 임베딩으로 검색

LLM 호출이 필요한 전략은 OpenAI API 또는 로컬 규칙 기반 fallback을 지원합니다.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class RewriteStrategy(Enum):
    """쿼리 리라이팅 전략."""

    NONE = "none"  # 리라이팅 안 함
    SIMPLE = "simple"  # 규칙 기반 + LLM 간단 변환
    MULTI_QUERY = "multi_query"  # 여러 변형으로 확장
    HYDE = "hyde"  # 가상 답변 생성 후 그것으로 검색


# --- 규칙 기반 리라이팅 (LLM 없이 동작) ---

# 약어 → 풀어쓰기 매핑
ABBREVIATION_MAP = {
    "RAG": "Retrieval-Augmented Generation (검색 증강 생성)",
    "LLM": "Large Language Model (대규모 언어 모델)",
    "VLM": "Vision Language Model (비전 언어 모델)",
    "ANN": "Approximate Nearest Neighbor (근사 최근접 이웃)",
    "HNSW": "Hierarchical Navigable Small World (계층적 탐색 소세계 그래프)",
    "CoT": "Chain of Thought (사고의 연쇄)",
    "OCR": "Optical Character Recognition (광학 문자 인식)",
    "CI": "Continuous Integration (지속적 통합)",
    "CD": "Continuous Deployment (지속적 배포)",
    "gRPC": "gRPC Remote Procedure Call",
}

# 불용어 / 질문 접두사 제거 패턴
NOISE_PATTERNS = [
    r"^(알려줘|설명해줘|말해줘|가르쳐줘)[.?!]?\s*",
    r"^(뭐야|뭘까|무엇인가요?|어떤 건가요?)[.?!]?\s*",
    r"^(혹시|그냥|좀|제발)\s+",
]


class QueryRewriter:
    """검색 최적화를 위한 쿼리 리라이터.

    LLM이 설정되지 않은 경우 규칙 기반 리라이팅만 수행합니다.
    OpenAI API key가 설정되면 LLM 기반 고급 리라이팅을 활성화합니다.
    """

    def __init__(self, openai_api_key: str | None = None, model: str = "gpt-4o-mini"):
        self._openai_api_key = openai_api_key
        self._model = model
        self._llm = None

    @property
    def llm_available(self) -> bool:
        """LLM 사용 가능 여부."""
        return bool(self._openai_api_key)

    def _get_llm(self):
        """Lazy-init OpenAI client."""
        if self._llm is None and self._openai_api_key:
            from openai import OpenAI

            self._llm = OpenAI(api_key=self._openai_api_key)
        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """LLM 호출 헬퍼."""
        llm = self._get_llm()
        if llm is None:
            return ""

        try:
            response = llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    # --- Simple Rewrite ---

    def rewrite(self, query: str) -> str:
        """검색에 최적화된 쿼리로 변환.

        LLM 사용 가능 시: LLM으로 리라이팅
        LLM 없을 시: 규칙 기반 변환 (약어 전개, 불용어 제거)
        """
        if self.llm_available:
            return self._rewrite_with_llm(query)
        return self._rewrite_rule_based(query)

    def _rewrite_rule_based(self, query: str) -> str:
        """규칙 기반 쿼리 리라이팅."""
        result = query.strip()

        # 불용어/질문 접두사 제거
        for pattern in NOISE_PATTERNS:
            result = re.sub(pattern, "", result)

        # 약어 전개 (괄호로 보충)
        for abbr, full in ABBREVIATION_MAP.items():
            # 영어 약어가 독립적으로 나타나는 경우 매칭 (한글 인접 허용)
            pattern = rf"(?<![A-Za-z]){re.escape(abbr)}(?![A-Za-z])"
            if re.search(pattern, result) and full not in result:
                result = re.sub(pattern, f"{abbr}({full})", result, count=1)

        return result.strip() if result.strip() else query

    def _rewrite_with_llm(self, query: str) -> str:
        """LLM 기반 쿼리 리라이팅."""
        system = (
            "당신은 검색 쿼리 최적화 전문가입니다. "
            "사용자의 질문을 벡터 검색에 최적화된 형태로 변환하세요.\n"
            "규칙:\n"
            "- 핵심 키워드를 유지하면서 모호한 표현을 구체화\n"
            "- 약어가 있으면 풀어쓰기 추가\n"
            "- 질문형보다 서술형이 검색에 유리\n"
            "- 원래 의미를 변경하지 않기\n"
            "- 변환된 쿼리만 출력 (설명 없이)"
        )
        result = self._call_llm(system, query)
        return result if result else self._rewrite_rule_based(query)

    # --- Multi-Query Expansion ---

    def expand(self, query: str, n: int = 3) -> list[str]:
        """쿼리를 여러 변형으로 확장.

        LLM 사용 가능 시: 다양한 관점의 쿼리 생성
        LLM 없을 시: 원본 + 규칙 기반 변형 반환
        """
        if self.llm_available:
            return self._expand_with_llm(query, n)
        return self._expand_rule_based(query, n)

    def _expand_rule_based(self, query: str, n: int = 3) -> list[str]:
        """규칙 기반 쿼리 확장."""
        variants = [query]

        # 변형 1: 약어 전개 버전
        rewritten = self._rewrite_rule_based(query)
        if rewritten != query:
            variants.append(rewritten)

        # 변형 2: 키워드 추출 (조사 제거)
        keywords = re.sub(r"[은는이가을를의에서로와과도만]", " ", query)
        keywords = " ".join(keywords.split())
        if keywords != query:
            variants.append(keywords)

        return variants[:n]

    def _expand_with_llm(self, query: str, n: int = 3) -> list[str]:
        """LLM 기반 쿼리 확장."""
        system = (
            "당신은 검색 쿼리 확장 전문가입니다. "
            f"사용자의 질문을 {n}가지 다른 관점/표현으로 변형하세요.\n"
            "규칙:\n"
            "- 각 변형은 원본과 의미는 같지만 다른 키워드/표현 사용\n"
            "- 한 줄에 하나씩 출력\n"
            "- 번호나 설명 없이 변형된 쿼리만 출력"
        )
        result = self._call_llm(system, query)
        if not result:
            return self._expand_rule_based(query, n)

        variants = [line.strip() for line in result.split("\n") if line.strip()]
        # 원본도 포함
        if query not in variants:
            variants.insert(0, query)
        return variants[: n + 1]  # 원본 + n개 변형

    # --- HyDE (Hypothetical Document Embeddings) ---

    def hyde(self, query: str) -> str:
        """가상 답변을 생성하여 검색 쿼리로 사용.

        LLM 필수. LLM 없으면 원본 쿼리를 그대로 반환합니다.
        """
        if not self.llm_available:
            logger.info("HyDE requires LLM. Returning original query.")
            return query
        return self._hyde_with_llm(query)

    def _hyde_with_llm(self, query: str) -> str:
        """LLM으로 가상 문서 생성."""
        system = (
            "당신은 기술 문서 작성 전문가입니다. "
            "사용자의 질문에 대한 답변이 포함된 가상의 문서 단락을 작성하세요.\n"
            "규칙:\n"
            "- 200자 이내의 짧은 단락\n"
            "- 실제 문서처럼 사실적인 톤\n"
            "- 질문의 핵심 개념과 관련 키워드를 풍부하게 포함\n"
            "- 답변이 정확하지 않아도 됨 (검색용이므로 키워드 다양성이 중요)"
        )
        result = self._call_llm(system, query)
        return result if result else query
