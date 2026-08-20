"""벡터 검색 + 리랭킹 파이프라인 (텍스트 + 이미지 멀티모달)."""

import logging

import numpy as np
from pymilvus import Collection, connections

from src.config.settings import settings
from src.retrieval.query_rewriter import QueryRewriter, RewriteStrategy

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self):
        self._triton = None
        connections.connect(host=settings.milvus_host, port=settings.milvus_port)
        self.text_col = Collection(settings.text_collection)
        self.image_col = Collection(settings.image_collection)
        self.text_col.load()
        self.image_col.load()

        # Query rewriter 초기화
        self._rewrite_strategy = RewriteStrategy(settings.rewrite_strategy)
        self._rewriter = QueryRewriter(
            openai_api_key=settings.rewrite_openai_api_key or None,
            model=settings.rewrite_llm_model,
        )

        # Embedding model (local fallback)
        self._embed_model = None

    @property
    def triton(self):
        """Lazy-init Triton client."""
        if self._triton is None and settings.use_triton:
            import tritonclient.grpc as grpcclient

            self._triton = grpcclient.InferenceServerClient(url=settings.triton_url)
        return self._triton

    @property
    def embed_model(self):
        """Lazy-init local embedding model."""
        if self._embed_model is None:
            from src.models.embedding import EmbeddingModel

            self._embed_model = EmbeddingModel()
        return self._embed_model

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """텍스트 전용 검색 (query rewriting 포함)."""
        # Query rewriting 단계
        if self._rewrite_strategy == RewriteStrategy.MULTI_QUERY:
            return self._search_multi_query(query, top_k)
        elif self._rewrite_strategy == RewriteStrategy.HYDE:
            search_query = self._rewriter.hyde(query)
            logger.debug(f"HyDE rewrite: '{query}' → '{search_query[:100]}...'")
        elif self._rewrite_strategy == RewriteStrategy.SIMPLE:
            search_query = self._rewriter.rewrite(query)
            logger.debug(f"Simple rewrite: '{query}' → '{search_query}'")
        else:
            search_query = query

        return self._search_single(search_query, top_k)

    def _search_single(self, query: str, top_k: int = 10) -> list[dict]:
        """단일 쿼리로 검색 (내부 핵심 로직)."""
        query_embedding = self._embed_query(query)

        text_results = self.text_col.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k * 2,
            output_fields=["text", "doc_id", "page_num"],
        )

        candidates = [hit.entity.get("text") for hit in text_results[0]]
        if not candidates:
            return []

        reranked = self._rerank(query, candidates)

        results = []
        for idx, score in reranked[:top_k]:
            hit = text_results[0][idx]
            results.append(
                {
                    "text": candidates[idx],
                    "score": float(score),
                    "doc_id": hit.entity.get("doc_id"),
                    "page_num": hit.entity.get("page_num"),
                    "type": "text",
                }
            )
        return results

    def _search_multi_query(self, query: str, top_k: int = 10) -> list[dict]:
        """Multi-query: 여러 변형 쿼리로 검색 후 결과 병합."""
        expanded_queries = self._rewriter.expand(query, n=3)
        logger.debug(f"Multi-query expansion: {expanded_queries}")

        # 각 변형 쿼리로 검색
        all_results: dict[str, dict] = {}  # text → result dict (중복 제거용)
        for q in expanded_queries:
            results = self._search_single(q, top_k=top_k)
            for r in results:
                key = r["text"]
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # 점수순 정렬 후 top_k 반환
        merged = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def search_images(self, query: str, top_k: int = 10) -> list[dict]:
        """이미지 컬렉션에서 SigLIP 텍스트 인코더를 사용한 검색."""
        query_embedding = self._embed_query_for_images(query)

        image_results = self.image_col.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            output_fields=["image_path", "doc_id", "caption"],
        )

        results = []
        for hit in image_results[0]:
            results.append(
                {
                    "image_path": hit.entity.get("image_path"),
                    "score": float(hit.score),
                    "doc_id": hit.entity.get("doc_id"),
                    "caption": hit.entity.get("caption"),
                    "type": "image",
                }
            )
        return results

    def search_multimodal(self, query: str, top_k: int = 10) -> list[dict]:
        """텍스트 + 이미지 통합 멀티모달 검색. 점수 기준 정렬."""
        text_results = self.search(query, top_k=top_k)
        image_results = self.search_images(query, top_k=top_k)

        # 두 결과를 합쳐서 score 기준 내림차순 정렬
        merged = text_results + image_results
        merged.sort(key=lambda x: x["score"], reverse=True)

        return merged[:top_k]

    def _embed_query(self, query: str) -> list[float]:
        """BGE-M3로 텍스트 쿼리 임베딩 생성. Triton → 로컬 fallback."""
        if settings.use_triton and self.triton:
            try:
                import tritonclient.grpc as grpcclient

                input_tensor = grpcclient.InferInput("text", [1, 1], "BYTES")
                input_tensor.set_data_from_numpy(np.array([[query.encode()]], dtype=object))
                result = self.triton.infer("bge-m3", [input_tensor])
                return result.as_numpy("embedding")[0].tolist()
            except Exception as e:
                logger.warning(f"Triton embed failed, using local model: {e}")

        # Local fallback
        embeddings = self.embed_model.encode([query])
        return embeddings[0].tolist()

    def _embed_query_for_images(self, query: str) -> list[float]:
        """SigLIP 텍스트 인코더로 이미지 검색용 쿼리 임베딩 생성."""
        if settings.use_triton and self.triton:
            try:
                import tritonclient.grpc as grpcclient

                input_tensor = grpcclient.InferInput("text", [1, 1], "BYTES")
                input_tensor.set_data_from_numpy(np.array([[query.encode()]], dtype=object))
                result = self.triton.infer("siglip", [input_tensor])
                return result.as_numpy("embedding")[0].tolist()
            except Exception as e:
                logger.warning(f"Triton SigLIP failed: {e}")

        # Fallback: use text embedding (not ideal but functional for testing)
        return self._embed_query(query)

    def _rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """BGE-Reranker로 텍스트 후보 리랭킹. Triton → cosine similarity fallback."""
        if settings.use_triton and self.triton:
            try:
                import tritonclient.grpc as grpcclient

                n = len(passages)
                q_input = grpcclient.InferInput("query", [n, 1], "BYTES")
                q_input.set_data_from_numpy(np.array([[query.encode()]] * n, dtype=object))
                p_input = grpcclient.InferInput("passage", [n, 1], "BYTES")
                p_input.set_data_from_numpy(
                    np.array([[p.encode()] for p in passages], dtype=object)
                )

                result = self.triton.infer("bge-reranker", [q_input, p_input])
                scores = result.as_numpy("score").flatten()
                return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            except Exception as e:
                logger.warning(f"Triton rerank failed, using cosine fallback: {e}")

        # Fallback: cosine similarity 기반 재정렬
        query_emb = np.array(self._embed_query(query))
        passage_embs = self.embed_model.encode(passages)
        scores = passage_embs @ query_emb
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
