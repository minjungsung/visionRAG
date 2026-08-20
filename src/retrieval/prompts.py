"""프롬프트 템플릿 시스템.

질문 유형별 프롬프트 자동 선택, citation 강제, CoT 옵션 등을 제공합니다.
LangSmith에서 프롬프트 버전별 성능 비교가 가능합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """질문 유형 분류."""

    FACTUAL = "factual"  # 사실 확인형 ("~는 무엇인가요?")
    ANALYTICAL = "analytical"  # 분석형 ("왜 ~인가요?", "~의 장단점은?")
    COMPARATIVE = "comparative"  # 비교형 ("A vs B 차이점은?")
    HOW_TO = "how_to"  # 방법 질문 ("~하려면 어떻게 해야 하나요?")
    DEFAULT = "default"  # 기본 (분류 불가 시)


@dataclass
class PromptConfig:
    """프롬프트 설정."""

    system_prompt: str
    user_template: str
    require_citation: bool = True
    chain_of_thought: bool = False
    few_shot_examples: list[dict] | None = None
    query_type: QueryType = QueryType.DEFAULT

    def format_prompt(self, query: str, context: str) -> str:
        """최종 프롬프트를 생성합니다."""
        parts = []

        # System prompt
        parts.append(self.system_prompt)

        # Few-shot examples
        if self.few_shot_examples:
            parts.append("\n[예시]")
            for ex in self.few_shot_examples:
                parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")

        # User template with context and query
        user_content = self.user_template.format(context=context, query=query)
        parts.append(user_content)

        # Chain of thought instruction
        if self.chain_of_thought:
            parts.append("단계별로 생각하여 답변하세요.")

        # Citation instruction
        if self.require_citation:
            parts.append("반드시 답변의 근거가 되는 문서 내용을 [출처]로 표시하세요.")

        return "\n\n".join(parts)


# --- 프롬프트 템플릿 정의 ---

PROMPT_TEMPLATES: dict[QueryType, PromptConfig] = {
    QueryType.DEFAULT: PromptConfig(
        system_prompt="당신은 문서 기반 질의응답 전문가입니다. 제공된 문서만을 참고하여 정확하게 답변하세요.",
        user_template="[문서]\n{context}\n\n[질문]\n{query}\n\n[답변]",
        require_citation=True,
        chain_of_thought=False,
        query_type=QueryType.DEFAULT,
    ),
    QueryType.FACTUAL: PromptConfig(
        system_prompt=(
            "당신은 정확한 사실 확인 전문가입니다. "
            "제공된 문서에 근거하여 간결하고 정확하게 답변하세요. "
            "문서에 없는 내용은 '해당 정보를 문서에서 찾을 수 없습니다'라고 답하세요."
        ),
        user_template="[문서]\n{context}\n\n[질문]\n{query}\n\n[답변]",
        require_citation=True,
        chain_of_thought=False,
        query_type=QueryType.FACTUAL,
    ),
    QueryType.ANALYTICAL: PromptConfig(
        system_prompt=(
            "당신은 분석 전문가입니다. "
            "제공된 문서를 기반으로 원인, 이유, 장단점 등을 논리적으로 분석하세요. "
            "주장에는 반드시 문서의 근거를 포함하세요."
        ),
        user_template="[문서]\n{context}\n\n[질문]\n{query}\n\n[분석]",
        require_citation=True,
        chain_of_thought=True,
        query_type=QueryType.ANALYTICAL,
    ),
    QueryType.COMPARATIVE: PromptConfig(
        system_prompt=(
            "당신은 비교 분석 전문가입니다. "
            "제공된 문서를 기반으로 대상 간의 공통점과 차이점을 구조적으로 정리하세요. "
            "가능하면 표 형식으로 비교하세요."
        ),
        user_template="[문서]\n{context}\n\n[비교 질문]\n{query}\n\n[비교 분석]",
        require_citation=True,
        chain_of_thought=True,
        query_type=QueryType.COMPARATIVE,
        few_shot_examples=[
            {
                "question": "HNSW와 IVF 인덱스의 차이점은?",
                "answer": (
                    "| 항목 | HNSW | IVF |\n"
                    "|------|------|-----|\n"
                    "| 검색 속도 | 빠름 | 중간 |\n"
                    "| 메모리 사용 | 높음 | 낮음 |\n"
                    "| 빌드 시간 | 김 | 짧음 |\n"
                    "[출처: 인덱스 설정 문서]"
                ),
            }
        ],
    ),
    QueryType.HOW_TO: PromptConfig(
        system_prompt=(
            "당신은 기술 가이드 전문가입니다. "
            "제공된 문서를 기반으로 단계별로 명확하게 방법을 설명하세요. "
            "코드 예시가 있으면 포함하세요."
        ),
        user_template="[문서]\n{context}\n\n[방법 질문]\n{query}\n\n[단계별 가이드]",
        require_citation=True,
        chain_of_thought=True,
        query_type=QueryType.HOW_TO,
    ),
}


class PromptManager:
    """프롬프트 템플릿 매니저.

    질문 유형에 따라 적절한 프롬프트를 선택하고 포맷팅합니다.
    """

    def __init__(self, templates: dict[QueryType, PromptConfig] | None = None):
        self.templates = templates or PROMPT_TEMPLATES

    def get_config(self, query_type: QueryType | str | None = None) -> PromptConfig:
        """질문 유형에 맞는 PromptConfig를 반환합니다."""
        if query_type is None:
            return self.templates[QueryType.DEFAULT]

        if isinstance(query_type, str):
            try:
                query_type = QueryType(query_type)
            except ValueError:
                return self.templates[QueryType.DEFAULT]

        return self.templates.get(query_type, self.templates[QueryType.DEFAULT])

    def build_prompt(
        self,
        query: str,
        context: str,
        query_type: QueryType | str | None = None,
    ) -> str:
        """최종 프롬프트 문자열을 생성합니다."""
        config = self.get_config(query_type)
        return config.format_prompt(query, context)

    def list_types(self) -> list[str]:
        """사용 가능한 질문 유형 목록을 반환합니다."""
        return [qt.value for qt in self.templates]
