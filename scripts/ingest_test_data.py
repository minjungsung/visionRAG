"""테스트용 문서 인제스천 스크립트.

golden_qa.jsonl의 contexts를 Milvus에 임베딩하여 저장합니다.
로컬 SentenceTransformer 사용 (Triton 불필요).
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import Collection, connections

from src.config.settings import settings
from src.models.embedding import EmbeddingModel

GOLDEN_QA_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_qa.jsonl"


def main():
    # Connect to Milvus
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    col = Collection(settings.text_collection)

    # Load embedding model (local)
    print("Loading embedding model...")
    model = EmbeddingModel()

    # Load golden QA data
    contexts = []
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            for ctx in data.get("contexts", []):
                if ctx and ctx not in contexts:  # 중복 제거
                    contexts.append(ctx)

    print(f"Unique contexts to ingest: {len(contexts)}")

    if not contexts:
        print("No contexts found.")
        return

    # Embed contexts
    print("Computing embeddings...")
    embeddings = model.encode(contexts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Insert into Milvus
    ids = [str(uuid.uuid4()) for _ in contexts]
    doc_ids = ["golden_qa"] * len(contexts)
    page_nums = list(range(len(contexts)))

    data = [
        ids,
        doc_ids,
        contexts,
        embeddings.tolist(),
        page_nums,
    ]

    col.insert(data)
    col.flush()
    print(f"Inserted {len(contexts)} chunks into {settings.text_collection}")

    # Verify
    col.load()
    print(f"Collection count: {col.num_entities}")


if __name__ == "__main__":
    main()
