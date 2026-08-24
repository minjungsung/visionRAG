"""data/raw 폴더의 모든 .md 파일을 인제스천."""
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()
for f in sorted(Path("data/raw").glob("*.md")):
    doc_id = pipeline.ingest_file(f.name, f.read_bytes())
    print(f"done: {f.name} -> {doc_id}")
print("완료!")
