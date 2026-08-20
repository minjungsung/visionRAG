"""RAG 답변 생성 파이프라인."""

import logging
import time

import numpy as np

from src.config.settings import settings
from src.mlops import trace_rag_pipeline
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.prompts import PromptManager, QueryType

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.retriever = RetrievalPipeline()
        self.prompt_manager = PromptManager()
        self._triton = None
        self._openai = None

    @property
    def triton(self):
        """Lazy-init Triton client."""
        if self._triton is None and settings.use_triton:
            import tritonclient.grpc as grpcclient

            self._triton = grpcclient.InferenceServerClient(url=settings.triton_url)
        return self._triton

    @property
    def openai(self):
        """Lazy-init OpenAI client."""
        if self._openai is None and settings.openai_api_key:
            from openai import OpenAI

            self._openai = OpenAI(api_key=settings.openai_api_key)
        return self._openai

    def answer(
        self,
        query: str,
        top_k: int = 5,
        query_type: QueryType | str | None = None,
    ) -> dict:
        t0 = time.perf_counter()

        # Retrieval
        t_ret = time.perf_counter()
        results = self.retriever.search(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        if not results:
            return {"answer": "관련 문서를 찾지 못했습니다.", "sources": []}

        # Build prompt using PromptManager
        context = "\n\n---\n\n".join(r["text"] for r in results)
        prompt = self.prompt_manager.build_prompt(
            query=query,
            context=context,
            query_type=query_type,
        )

        # Generation: Triton → OpenAI fallback
        t_gen = time.perf_counter()
        answer_text = self._generate(prompt)
        generation_ms = (time.perf_counter() - t_gen) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        # LangSmith trace
        trace_rag_pipeline(
            query=query,
            results=results,
            answer=answer_text,
            timings={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
        )

        return {"answer": answer_text, "sources": results}

    def _generate(self, prompt: str) -> str:
        """답변 생성. Triton 먼저 시도, 실패 시 OpenAI fallback."""
        # 1) Triton (GPU)
        if settings.use_triton:
            try:
                return self._generate_triton(prompt)
            except Exception as e:
                logger.warning(f"Triton generation failed, falling back to OpenAI: {e}")

        # 2) OpenAI fallback
        if self.openai:
            return self._generate_openai(prompt)

        # 3) 둘 다 안 되면
        return "[답변 생성 불가] Triton 서버 또는 OpenAI API key가 필요합니다."

    def _generate_triton(self, prompt: str) -> str:
        """Triton으로 Qwen2-VL 답변 생성."""
        import tritonclient.grpc as grpcclient

        prompt_input = grpcclient.InferInput("prompt", [1, 1], "BYTES")
        prompt_input.set_data_from_numpy(np.array([[prompt.encode()]], dtype=object))
        image_input = grpcclient.InferInput("image", [1, 1], "UINT8")
        image_input.set_data_from_numpy(np.zeros((1, 1), dtype=np.uint8))

        result = self.triton.infer("qwen2-vl", [prompt_input, image_input])
        return result.as_numpy("response").flatten()[0].decode("utf-8")

    def _generate_openai(self, prompt: str) -> str:
        """OpenAI API로 답변 생성."""
        try:
            response = self.openai.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "문서 기반으로 정확하게 답변하세요."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"[답변 생성 실패] {e}"
