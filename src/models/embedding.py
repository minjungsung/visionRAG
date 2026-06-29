import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.settings import settings


class EmbeddingModel:
    def __init__(self):
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer("BAAI/bge-m3")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)


embedding_model = EmbeddingModel()
