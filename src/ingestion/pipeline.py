"""문서 인제스천 파이프라인: 파싱 → 임베딩 → Milvus 저장."""

import base64
import io
import logging
import uuid

from minio import Minio
from PIL import Image
from pymilvus import Collection, connections

from src.config.settings import settings
from src.models.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.embed_model = EmbeddingModel()
        self.minio = Minio(
            settings.minio_endpoint,
            settings.minio_access_key,
            settings.minio_secret_key,
            secure=False,
        )
        connections.connect(host=settings.milvus_host, port=settings.milvus_port)
        self.text_col = Collection(settings.text_collection)
        self.image_col = Collection(settings.image_collection)

    def ingest_file(self, file_path: str, file_bytes: bytes) -> str:
        doc_id = str(uuid.uuid4())

        # MinIO에 원본 저장
        self._store_to_minio(doc_id, file_path, file_bytes)

        # 파싱
        elements = self._parse(file_path, file_bytes)

        texts, images = [], []
        for el in elements:
            if el.category == "Image" and getattr(el.metadata, "image_base64", None):
                images.append(el)
            elif hasattr(el, "text") and el.text.strip():
                texts.append(el)

        if texts:
            self._ingest_texts(doc_id, texts)
        if images:
            self._ingest_images(doc_id, images)

        logger.info(f"Ingested {file_path}: {len(texts)} texts, {len(images)} images → {doc_id}")
        return doc_id

    def _store_to_minio(self, doc_id: str, file_path: str, file_bytes: bytes) -> None:
        """원본 파일을 MinIO에 저장."""
        bucket = settings.minio_bucket
        if not self.minio.bucket_exists(bucket):
            self.minio.make_bucket(bucket)
        self.minio.put_object(
            bucket, f"{doc_id}/{file_path}", io.BytesIO(file_bytes), len(file_bytes)
        )

    def _parse(self, file_path: str, file_bytes: bytes):
        """unstructured로 문서 파싱. 설치 안 되어 있으면 간단한 텍스트 분할."""
        try:
            from unstructured.partition.auto import partition

            return partition(file=io.BytesIO(file_bytes), metadata_filename=file_path)
        except ImportError:
            logger.warning("unstructured not installed, using simple text split")
            return self._simple_parse(file_path, file_bytes)

    def _simple_parse(self, file_path: str, file_bytes: bytes):
        """unstructured 없을 때 간단한 텍스트 파싱."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeMetadata:
            page_number: int = 0

        @dataclass
        class FakeElement:
            text: str = ""
            category: str = "Text"
            metadata: FakeMetadata = field(default_factory=FakeMetadata)

        text = file_bytes.decode("utf-8", errors="replace")
        # 단락 단위로 분할
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [FakeElement(text=p) for p in paragraphs]

    def _ingest_texts(self, doc_id: str, elements) -> None:
        """텍스트 청크 임베딩 생성 후 Milvus 저장."""
        chunks = [el.text for el in elements]

        # EmbeddingModel 사용 (Triton/local 자동 선택)
        embeddings = self.embed_model.encode(chunks)

        self.text_col.insert(
            [
                [str(uuid.uuid4()) for _ in chunks],
                [doc_id] * len(chunks),
                chunks,
                embeddings.tolist(),
                [getattr(el.metadata, "page_number", 0) or 0 for el in elements],
            ]
        )
        logger.debug(f"Inserted {len(chunks)} text chunks for doc {doc_id}")

    def _ingest_images(self, doc_id: str, elements) -> None:
        """이미지 임베딩 생성 후 Milvus 저장."""
        for el in elements:
            img_bytes = base64.b64decode(el.metadata.image_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # MinIO에 이미지 저장
            img_key = f"{doc_id}/images/{uuid.uuid4()}.png"
            self.minio.put_object(
                settings.minio_bucket, img_key, io.BytesIO(img_bytes), len(img_bytes)
            )

            # EmbeddingModel 사용 (Triton/local 자동 선택)
            embedding = self.embed_model.encode_image([img])

            self.image_col.insert(
                [
                    [str(uuid.uuid4())],
                    [doc_id],
                    [img_key],
                    embedding.tolist(),
                    [""],
                ]
            )
        logger.debug(f"Inserted {len(elements)} images for doc {doc_id}")
