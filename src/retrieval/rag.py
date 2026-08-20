"""RAG 답변 생성 파이프라인."""

import time

import numpy as np
import tritonclient.grpc as grpcclient

from src.config.settings import settings
from src.mlops import trace_rag_pipeline
from src.retrieval.pipeline import RetrievalPipeline


class RAGPipeline:
    def __init__(self):
        self.retriever = RetrievalPipeline()
        self.triton = grpcclient.InferenceServerClient(url=settings.triton_url)

    def answer(self, query: str, top_k: int = 5) -> dict:
        t0 = time.perf_counter()

        # Retrieval
        t_ret = time.perf_counter()
        results = self.retriever.search(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        if not results:
            return {"answer": "관련 문서를 찾지 못했습니다.", "sources": []}

        # Generation
        context = "\n\n---\n\n".join(r["text"] for r in results)
        prompt = (
            f"다음 문서를 참고하여 질문에 답하세요.\n\n"
            f"[문서]\n{context}\n\n"
            f"[질문]\n{query}\n\n"
            f"[답변]"
        )

        t_gen = time.perf_counter()
        prompt_input = grpcclient.InferInput("prompt", [1, 1], "BYTES")
        prompt_input.set_data_from_numpy(np.array([[prompt.encode()]], dtype=object))
        image_input = grpcclient.InferInput("image", [1, 1], "UINT8")
        image_input.set_data_from_numpy(np.zeros((1, 1), dtype=np.uint8))

        result = self.triton.infer("qwen2-vl", [prompt_input, image_input])
        answer_text = result.as_numpy("response").flatten()[0].decode("utf-8")
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
