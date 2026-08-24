"""벡터 검색 파이프라인 (FAISS 기반)."""

import logging
import time

from src.models.embedding import EmbeddingModel
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self):
        self.embed_model = EmbeddingModel()
        self.store = VectorStore()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """텍스트 검색."""
        query_embedding = self.embed_model.encode([query])
        results = self.store.search(query_embedding[0], top_k=top_k)
        return results

    def search_multimodal(self, query: str, top_k: int = 5) -> list[dict]:
        """멀티모달 검색 (현재는 텍스트만)."""
        return self.search(query, top_k=top_k)
