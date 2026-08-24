"""FAISS 기반 벡터 스토어 — Milvus 없이 동작.

로컬 + HuggingFace Spaces 둘 다 사용 가능.
데이터는 .index 파일 + chunks.jsonl로 저장.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/index")


class VectorStore:
    """FAISS 기반 벡터 검색 엔진."""

    def __init__(self, index_dir: Path | str = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self._loaded = False

    def load(self) -> None:
        """저장된 인덱스 로드."""
        if self._loaded:
            return

        index_path = self.index_dir / "faiss.index"
        chunks_path = self.index_dir / "chunks.jsonl"

        if not index_path.exists():
            logger.warning(f"인덱스 파일 없음: {index_path}. build() 먼저 실행하세요.")
            return

        self.index = faiss.read_index(str(index_path))
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))

        self._loaded = True
        logger.info(f"로드 완료: {len(self.chunks)}개 청크, dim={self.index.d}")

    def build(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """청크 + 임베딩으로 인덱스 빌드 및 저장."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # FAISS 인덱스 생성 (Inner Product = Cosine similarity for normalized vectors)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks

        # 저장
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write("\n")

        self._loaded = True
        logger.info(f"빌드 완료: {len(chunks)}개 청크, dim={dim}")

    def add(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """기존 인덱스에 데이터 추가."""
        self.load()
        if self.index is None:
            self.build(chunks, embeddings)
            return

        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

        # 재저장
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.jsonl", "a", encoding="utf-8") as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"추가 완료: +{len(chunks)}개, 총 {len(self.chunks)}개")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """벡터 검색."""
        self.load()
        if self.index is None or len(self.chunks) == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    @property
    def count(self) -> int:
        """저장된 청크 수."""
        self.load()
        return len(self.chunks) if self.chunks else 0
