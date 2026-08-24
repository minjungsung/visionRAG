"""RAG 답변 생성 파이프라인."""

import logging
import time

from src.config.settings import settings
from src.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.retriever = RetrievalPipeline()
        self._openai = None

    @property
    def openai(self):
        """Lazy-init OpenAI client."""
        if self._openai is None and settings.openai_api_key:
            from openai import OpenAI

            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            self._openai = OpenAI(**kwargs)
        return self._openai

    def answer(self, query: str, top_k: int = 5, query_type=None) -> dict:
        t0 = time.perf_counter()

        # 검색
        results = self.retriever.search(query, top_k=top_k)

        if not results:
            return {"answer": "관련 문서를 찾지 못했습니다.", "sources": []}

        # 컨텍스트 구성
        context = "\n\n---\n\n".join(r["text"] for r in results)

        # 답변 생성
        answer_text = self._generate(query, context)

        elapsed = time.perf_counter() - t0
        logger.info(f"RAG answer in {elapsed:.1f}s, {len(results)} sources")

        return {"answer": answer_text, "sources": results}

    def _generate(self, query: str, context: str) -> str:
        """LLM으로 답변 생성. OpenAI 없으면 검색 결과만 반환."""
        if self.openai:
            try:
                prompt = (
                    f"다음 문서를 참고하여 질문에 답하세요.\n\n"
                    f"[문서]\n{context}\n\n"
                    f"[질문]\n{query}\n\n"
                    f"[답변]"
                )
                response = self.openai.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "문서 기반으로 정확하게 한국어로 답변하세요.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI failed: {e}")

        # OpenAI 없으면 검색 결과 요약
        return f"[검색 결과 기반]\n\n" + "\n\n".join(
            f"• {r['text'][:200]}" for r in self.retriever.search(query, top_k=3)
        )
