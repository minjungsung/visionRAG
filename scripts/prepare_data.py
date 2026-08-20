"""DVC pipeline stage: prepare_data

Reads documents from data/raw/ (.txt, .pdf, .md), splits them into chunks,
and saves processed chunks to data/processed/ as JSONL files.
Outputs metrics to data/metrics.json.

No external services required (no Triton/Milvus).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_PATH = PROJECT_ROOT / "data" / "metrics.json"

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def read_text_file(path: Path) -> str:
    """Read a plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf_file(path: Path) -> str:
    """Read a PDF file. Uses PyMuPDF (fitz) if available, else pdfplumber, else raw bytes."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pass

    logger.warning(
        f"No PDF library available (install PyMuPDF or pdfplumber). Skipping {path.name}."
    )
    return ""


def read_document(path: Path) -> str:
    """Read document content based on file extension."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return read_text_file(path)
    elif suffix == ".pdf":
        return read_pdf_file(path)
    else:
        logger.warning(f"Unsupported file type: {path.name}")
        return ""


def recursive_character_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text recursively using a hierarchy of separators.

    Tries the first separator; if resulting pieces are still too large,
    recursively splits with the next separator.
    """
    if separators is None:
        separators = SEPARATORS

    if not text or len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator for this level
    separator = separators[-1]  # fallback: empty string (char-level)
    for sep in separators:
        if sep in text:
            separator = sep
            break

    remaining_separators = (
        separators[separators.index(separator) + 1 :] if separator in separators else separators[1:]
    )

    # Split on the chosen separator
    parts = text.split(separator) if separator else list(text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for part in parts:
        part_length = len(part) + (len(separator) if current_chunk else 0)

        if current_length + part_length > chunk_size and current_chunk:
            # Emit the current chunk
            merged = separator.join(current_chunk)
            if len(merged) > chunk_size and remaining_separators:
                # Recursively split if still too large
                chunks.extend(
                    recursive_character_split(
                        merged, chunk_size, chunk_overlap, remaining_separators
                    )
                )
            else:
                chunks.append(merged)

            # Overlap: keep trailing parts that fit within overlap budget
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_chunk):
                candidate_len = overlap_len + len(p) + (len(separator) if overlap_parts else 0)
                if candidate_len > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_len = candidate_len

            current_chunk = overlap_parts
            current_length = overlap_len

        current_chunk.append(part)
        current_length += part_length

    # Emit remaining
    if current_chunk:
        merged = separator.join(current_chunk)
        if len(merged) > chunk_size and remaining_separators:
            chunks.extend(
                recursive_character_split(merged, chunk_size, chunk_overlap, remaining_separators)
            )
        else:
            chunks.append(merged)

    # Filter empty chunks
    return [c for c in chunks if c.strip()]


def generate_doc_id(filepath: Path) -> str:
    """Generate a stable document ID from the file path relative to raw dir."""
    rel_path = filepath.relative_to(RAW_DIR)
    return hashlib.md5(str(rel_path).encode()).hexdigest()[:12]


def process_documents() -> Iterator[dict]:
    """Process all documents in data/raw/ and yield chunk records."""
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created empty {RAW_DIR}. Add documents and re-run.")
        return

    supported_extensions = {".txt", ".pdf", ".md"}
    files = sorted(
        f for f in RAW_DIR.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions
    )

    if not files:
        logger.warning(f"No supported documents found in {RAW_DIR}")
        return

    logger.info(f"Found {len(files)} documents in {RAW_DIR}")

    for filepath in files:
        doc_id = generate_doc_id(filepath)
        text = read_document(filepath)

        if not text.strip():
            logger.warning(f"Empty content for {filepath.name}, skipping.")
            continue

        chunks = recursive_character_split(text)
        rel_path = str(filepath.relative_to(RAW_DIR))

        for idx, chunk_text in enumerate(chunks):
            yield {
                "chunk_id": f"{doc_id}_{idx:04d}",
                "doc_id": doc_id,
                "source_file": rel_path,
                "chunk_index": idx,
                "text": chunk_text,
                "char_length": len(chunk_text),
            }


def main() -> None:
    """Main pipeline entry point."""
    # Ensure output directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Clean previous outputs
    for f in PROCESSED_DIR.glob("*.jsonl"):
        f.unlink()

    # Process documents
    total_chunks = 0
    total_length = 0
    num_files = 0
    current_source: str | None = None
    output_file = None

    try:
        for record in process_documents():
            # Group output by source file
            if record["source_file"] != current_source:
                if output_file is not None:
                    output_file.close()
                current_source = record["source_file"]
                num_files += 1
                # Create output JSONL named after the source
                safe_name = record["source_file"].replace("/", "__").rsplit(".", 1)[0]
                output_path = PROCESSED_DIR / f"{safe_name}.jsonl"
                output_file = open(output_path, "w", encoding="utf-8")
                logger.info(f"Processing: {record['source_file']} -> {output_path.name}")

            json.dump(record, output_file, ensure_ascii=False)
            output_file.write("\n")
            total_chunks += 1
            total_length += record["char_length"]

    finally:
        if output_file is not None:
            output_file.close()

    # Compute and save metrics
    avg_chunk_length = total_length / total_chunks if total_chunks > 0 else 0.0
    metrics = {
        "num_files": num_files,
        "num_chunks": total_chunks,
        "avg_chunk_length": round(avg_chunk_length, 2),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Output saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
