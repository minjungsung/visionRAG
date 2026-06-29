"""벡터 검색 + 리랭킹 파이프라인."""
import numpy as np
import tritonclient.grpc as grpcclient
from pymilvus import connections, Collection

from src.config.settings import settings


class RetrievalPipeline:
    def __init__(self):
        self.triton = grpcclient.InferenceServerClient(url=settings.triton_url)
        connections.connect(host=settings.milvus_host, port=settings.milvus_port)
        self.text_col = Collection(settings.text_collection)
        self.image_col = Collection(settings.image_collection)
        self.text_col.load()
        self.image_col.load()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
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
            results.append({
                "text": candidates[idx],
                "score": float(score),
                "doc_id": hit.entity.get("doc_id"),
                "page_num": hit.entity.get("page_num"),
            })
        return results

    def _embed_query(self, query: str) -> list[float]:
        input_tensor = grpcclient.InferInput("text", [1, 1], "BYTES")
        input_tensor.set_data_from_numpy(np.array([[query.encode()]], dtype=object))
        result = self.triton.infer("bge-m3", [input_tensor])
        return result.as_numpy("embedding")[0].tolist()

    def _rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        n = len(passages)
        q_input = grpcclient.InferInput("query", [n, 1], "BYTES")
        q_input.set_data_from_numpy(np.array([[query.encode()]] * n, dtype=object))
        p_input = grpcclient.InferInput("passage", [n, 1], "BYTES")
        p_input.set_data_from_numpy(np.array([[p.encode()] for p in passages], dtype=object))

        result = self.triton.infer("bge-reranker", [q_input, p_input])
        scores = result.as_numpy("score").flatten()
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
