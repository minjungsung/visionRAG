"""문서 인제스천 파이프라인: 파싱 → 임베딩 → Milvus 저장."""
import base64
import io
import uuid

import numpy as np
import tritonclient.grpc as grpcclient
from minio import Minio
from PIL import Image
from pymilvus import connections, Collection
from unstructured.partition.auto import partition

from src.config.settings import settings


class IngestionPipeline:
    def __init__(self):
        self.triton = grpcclient.InferenceServerClient(url=settings.triton_url)
        self.minio = Minio(
            settings.minio_endpoint, settings.minio_access_key, settings.minio_secret_key, secure=False
        )
        connections.connect(host=settings.milvus_host, port=settings.milvus_port)
        self.text_col = Collection(settings.text_collection)
        self.image_col = Collection(settings.image_collection)

    def ingest_file(self, file_path: str, file_bytes: bytes) -> str:
        doc_id = str(uuid.uuid4())

        # MinIO에 원본 저장
        self.minio.put_object(
            settings.minio_bucket, f"{doc_id}/{file_path}", io.BytesIO(file_bytes), len(file_bytes)
        )

        # unstructured로 파싱
        elements = partition(file=io.BytesIO(file_bytes), metadata_filename=file_path)

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

        return doc_id

    def _ingest_texts(self, doc_id: str, elements):
        chunks = [el.text for el in elements]
        input_tensor = grpcclient.InferInput("text", [len(chunks), 1], "BYTES")
        input_tensor.set_data_from_numpy(np.array([[c.encode()] for c in chunks], dtype=object))
        result = self.triton.infer("bge-m3", [input_tensor])
        embeddings = result.as_numpy("embedding")

        self.text_col.insert([
            [str(uuid.uuid4()) for _ in chunks],
            [doc_id] * len(chunks),
            chunks,
            embeddings.tolist(),
            [getattr(el.metadata, "page_number", 0) or 0 for el in elements],
        ])

    def _ingest_images(self, doc_id: str, elements):
        for el in elements:
            img_bytes = base64.b64decode(el.metadata.image_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((384, 384))
            pixel = np.array(img).transpose(2, 0, 1).astype(np.uint8)

            img_key = f"{doc_id}/images/{uuid.uuid4()}.png"
            self.minio.put_object(settings.minio_bucket, img_key, io.BytesIO(img_bytes), len(img_bytes))

            input_tensor = grpcclient.InferInput("image", [1, 3, 384, 384], "UINT8")
            input_tensor.set_data_from_numpy(pixel[np.newaxis, ...])
            result = self.triton.infer("siglip", [input_tensor])
            embedding = result.as_numpy("embedding")

            self.image_col.insert([
                [str(uuid.uuid4())], [doc_id], [img_key], embedding.tolist(), [""],
            ])
