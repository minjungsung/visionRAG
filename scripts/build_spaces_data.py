"""Spaces 배포용 임베딩 사전 계산.

data/raw/*.md → spaces/data/chunks.jsonl + spaces/data/embeddings.npz
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.embedding import EmbeddingModel

RAW_DIR = Path("data/raw")
OUT_DIR = Path("spaces/data")


def split_paragraphs(text: str, source: str) -> list[dict]:
    """텍스트를 단락 단위로 분리."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [{"text": p, "source": source} for p in paragraphs]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 모든 md 파일에서 청크 생성
    chunks = []
    for f in sorted(RAW_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        file_chunks = split_paragraphs(text, f.name)
        chunks.extend(file_chunks)
        print(f"  {f.name}: {len(file_chunks)}개 청크")

    print(f"\n총 {len(chunks)}개 청크")

    # 2. 임베딩 계산
    print("임베딩 계산 중...")
    model = EmbeddingModel()
    texts = [c["text"] for c in chunks]

    # 배치로 처리
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch)
        all_embeddings.append(emb)
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)
    print(f"임베딩 shape: {embeddings.shape}")

    # 3. 저장
    chunks_file = OUT_DIR / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in chunks:
            json.dump(c, f, ensure_ascii=False)
            f.write("\n")

    embeddings_file = OUT_DIR / "embeddings.npz"
    np.savez_compressed(embeddings_file, embeddings=embeddings)

    print(f"\n✅ 저장 완료:")
    print(f"  {chunks_file} ({chunks_file.stat().st_size / 1024:.1f} KB)")
    print(f"  {embeddings_file} ({embeddings_file.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
