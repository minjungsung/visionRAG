"""Milvus 컬렉션 초기화 스크립트."""
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

from src.config.settings import settings


def create_collections():
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    # 텍스트 컬렉션
    if not utility.has_collection(settings.text_collection):
        fields = [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=64),
            FieldSchema("text", DataType.VARCHAR, max_length=8192),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=settings.text_embed_dim),
            FieldSchema("page_num", DataType.INT64),
        ]
        col = Collection(settings.text_collection, CollectionSchema(fields))
        col.create_index(
            "embedding",
            {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}},
        )
        print(f"Created collection: {settings.text_collection}")

    # 이미지 컬렉션
    if not utility.has_collection(settings.image_collection):
        fields = [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=64),
            FieldSchema("image_path", DataType.VARCHAR, max_length=512),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=settings.image_embed_dim),
            FieldSchema("caption", DataType.VARCHAR, max_length=2048),
        ]
        col = Collection(settings.image_collection, CollectionSchema(fields))
        col.create_index(
            "embedding",
            {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}},
        )
        print(f"Created collection: {settings.image_collection}")

    print("Done.")


if __name__ == "__main__":
    create_collections()
