"""문서 인제스천 파이프라인: 파싱 → 임베딩 → FAISS 저장."""

import logging
import uuid
from pathlib import Path

from src.models.embedding import EmbeddingModel
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.embed_model = EmbeddingModel()
        self.store = VectorStore()

    def ingest_file(self, file_path: str, file_bytes: bytes) -> str:
        """파일을 파싱해서 벡터 스토어에 저장."""
        doc_id = str(uuid.uuid4())

        # 파싱
        elements = self._parse(file_path, file_bytes)

        texts = []
        for el in elements:
            if hasattr(el, "text") and el.text.strip():
                texts.append(el)

        if texts:
            self._ingest_texts(doc_id, file_path, texts)

        logger.info(f"Ingested {file_path}: {len(texts)} chunks → {doc_id}")
        return doc_id

    def _parse(self, file_path: str, file_bytes: bytes):
        """문서 파싱. unstructured 있으면 사용, 없으면 단순 텍스트 분할."""
        try:
            import io

            from unstructured.partition.auto import partition

            return partition(file=io.BytesIO(file_bytes), metadata_filename=file_path)
        except ImportError:
            logger.info("unstructured not installed, using simple text split")
            return self._simple_parse(file_bytes)

    def _simple_parse(self, file_bytes: bytes):
        """단순 텍스트 파싱 — 단락 단위 분리."""
        from dataclasses import dataclass

        @dataclass
        class FakeElement:
            text: str = ""
            category: str = "Text"

        text = file_bytes.decode("utf-8", errors="replace")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [FakeElement(text=p) for p in paragraphs]

    def _ingest_texts(self, doc_id: str, source: str, elements) -> None:
        """텍스트 임베딩 생성 후 FAISS에 저장."""
        chunks_data = []
        texts = []

        for el in elements:
            text = el.text.strip()
            if text:
                texts.append(text)
                chunks_data.append(
                    {
                        "text": text,
                        "doc_id": doc_id,
                        "source": source,
                    }
                )

        # 임베딩 생성
        embeddings = self.embed_model.encode(texts)

        # FAISS에 추가
        self.store.add(chunks_data, embeddings)
        logger.debug(f"Added {len(texts)} chunks for {source}")
