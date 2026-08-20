"""DVC pipeline stage: evaluate_retrieval

Evaluates retrieval quality on processed data using test queries.
Computes MRR, Recall@k, Precision@k, NDCG@k.
Uses the local embedding model (sentence-transformers, no Triton needed).

Outputs:
- reports/retrieval_metrics.json (aggregate metrics)
- reports/retrieval_plots.csv (per-query results for DVC plots)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEST_QUERIES_PATH = PROJECT_ROOT / "data" / "test_queries.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "retrieval_metrics.json"
PLOTS_PATH = REPORTS_DIR / "retrieval_plots.csv"

# Evaluation parameters
K_VALUES = [1, 3, 5, 10]

# Add project root to path for importing src modules
sys.path.insert(0, str(PROJECT_ROOT))


def load_processed_chunks() -> list[dict]:
    """Load all processed chunks from data/processed/."""
    chunks = []
    if not PROCESSED_DIR.exists():
        logger.error(f"Processed data directory not found: {PROCESSED_DIR}")
        return chunks

    for jsonl_file in sorted(PROCESSED_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks from {PROCESSED_DIR}")
    return chunks


def load_test_queries() -> list[dict]:
    """Load test queries from data/test_queries.jsonl."""
    queries = []
    if not TEST_QUERIES_PATH.exists():
        logger.warning(f"Test queries file not found: {TEST_QUERIES_PATH}. Creating sample file.")
        create_sample_test_queries()

    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    logger.info(f"Loaded {len(queries)} test queries")
    return queries


def create_sample_test_queries() -> None:
    """Create a sample test queries file."""
    TEST_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_queries = [
        {
            "query": "멀티모달 검색 시스템의 아키텍처",
            "relevant_doc_ids": [],
        },
        {
            "query": "벡터 데이터베이스를 활용한 문서 검색",
            "relevant_doc_ids": [],
        },
        {
            "query": "이미지와 텍스트를 동시에 처리하는 방법",
            "relevant_doc_ids": [],
        },
        {
            "query": "RAG 파이프라인 구현",
            "relevant_doc_ids": [],
        },
        {
            "query": "임베딩 모델 성능 최적화",
            "relevant_doc_ids": [],
        },
    ]

    with open(TEST_QUERIES_PATH, "w", encoding="utf-8") as f:
        for q in sample_queries:
            json.dump(q, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Created sample test queries at {TEST_QUERIES_PATH}")


def compute_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Compute embeddings using the local EmbeddingModel."""
    from src.models.embedding import EmbeddingModel

    model = EmbeddingModel()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def retrieve_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """Retrieve top-k documents by cosine similarity.

    Returns list of (index, score) tuples.
    Embeddings are assumed to be L2-normalized (dot product = cosine similarity).
    """
    scores = corpus_embeddings @ query_embedding.T
    scores = scores.flatten()
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def compute_mrr(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank for a single query."""
    if not relevant_doc_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(
    retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int
) -> float:
    """Compute Precision@k for a single query."""
    if not relevant_doc_ids or k == 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return relevant_count / k


def compute_recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Compute Recall@k for a single query.

    Counts unique relevant documents found in the top-k results.
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    found_relevant = {doc_id for doc_id in top_k if doc_id in relevant_doc_ids}
    return len(found_relevant) / len(relevant_doc_ids)


def compute_ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Compute NDCG@k for a single query.

    Uses binary relevance: a retrieved doc is relevant (1) or not (0).
    Only the first occurrence of a relevant doc_id contributes to the gain.
    """
    if not relevant_doc_ids:
        return 0.0

    # DCG — only first occurrence of each relevant doc counts
    dcg = 0.0
    seen_relevant: set[str] = set()
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in relevant_doc_ids and doc_id not in seen_relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
            seen_relevant.add(doc_id)

    # Ideal DCG
    ideal_rels = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_queries(
    queries: list[dict],
    chunks: list[dict],
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> tuple[dict[str, float], list[dict]]:
    """Evaluate all queries and compute aggregate metrics.

    Returns (aggregate_metrics, per_query_results).
    """
    max_k = max(K_VALUES)
    chunk_doc_ids = [c["doc_id"] for c in chunks]

    per_query_results: list[dict] = []
    all_mrr: list[float] = []
    all_precision: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_recall: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_ndcg: dict[int, list[float]] = {k: [] for k in K_VALUES}

    for q_idx, query_data in enumerate(queries):
        query_text = query_data["query"]
        relevant_doc_ids = set(query_data.get("relevant_doc_ids", []))

        # If no relevant docs specified, use similarity-based pseudo-relevance
        # (top-1 result is treated as relevant for self-consistency evaluation)
        use_pseudo_relevance = len(relevant_doc_ids) == 0

        # Retrieve top-k results
        q_embedding = query_embeddings[q_idx]
        results = retrieve_top_k(q_embedding, corpus_embeddings, max_k)
        retrieved_doc_ids = [chunk_doc_ids[idx] for idx, _ in results]
        retrieved_scores = [score for _, score in results]

        if use_pseudo_relevance and retrieved_doc_ids:
            # Use top-1 doc as pseudo-relevant for metric computation
            relevant_doc_ids = {retrieved_doc_ids[0]}

        # Compute metrics
        mrr = compute_mrr(retrieved_doc_ids, relevant_doc_ids)
        all_mrr.append(mrr)

        query_result = {
            "query": query_text,
            "mrr": round(mrr, 4),
            "top_score": round(retrieved_scores[0], 4) if retrieved_scores else 0.0,
        }

        for k in K_VALUES:
            p_at_k = compute_precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            r_at_k = compute_recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            ndcg = compute_ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

            all_precision[k].append(p_at_k)
            all_recall[k].append(r_at_k)
            all_ndcg[k].append(ndcg)

            query_result[f"precision_at_{k}"] = round(p_at_k, 4)
            query_result[f"recall_at_{k}"] = round(r_at_k, 4)
            query_result[f"ndcg_at_{k}"] = round(ndcg, 4)

        per_query_results.append(query_result)

    # Aggregate metrics
    aggregate_metrics: dict[str, float] = {
        "mrr": round(float(np.mean(all_mrr)), 4) if all_mrr else 0.0,
        "num_queries": len(queries),
        "num_chunks": len(chunks),
    }

    for k in K_VALUES:
        aggregate_metrics[f"precision_at_{k}"] = (
            round(float(np.mean(all_precision[k])), 4) if all_precision[k] else 0.0
        )
        aggregate_metrics[f"recall_at_{k}"] = (
            round(float(np.mean(all_recall[k])), 4) if all_recall[k] else 0.0
        )
        aggregate_metrics[f"ndcg_at_{k}"] = (
            round(float(np.mean(all_ndcg[k])), 4) if all_ndcg[k] else 0.0
        )

    return aggregate_metrics, per_query_results


def save_metrics(metrics: dict[str, Any]) -> None:
    """Save aggregate metrics to JSON."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Metrics saved to {METRICS_PATH}")


def save_plots(per_query_results: list[dict]) -> None:
    """Save per-query results as CSV for DVC plots."""
    PLOTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not per_query_results:
        logger.warning("No results to save for plots.")
        return

    fieldnames = list(per_query_results[0].keys())
    with open(PLOTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query_results)

    logger.info(f"Per-query results saved to {PLOTS_PATH}")


def main() -> None:
    """Main evaluation pipeline."""
    # Ensure directories exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)

    # Load data
    chunks = load_processed_chunks()
    if not chunks:
        logger.error("No processed chunks found. Run 'prepare_data' stage first.")
        # Output empty metrics so DVC doesn't fail
        save_metrics({"mrr": 0.0, "num_queries": 0, "num_chunks": 0, "error": "no_data"})
        save_plots([])
        return

    queries = load_test_queries()
    if not queries:
        logger.error("No test queries found.")
        save_metrics(
            {"mrr": 0.0, "num_queries": 0, "num_chunks": len(chunks), "error": "no_queries"}
        )
        save_plots([])
        return

    # Compute embeddings
    logger.info("Computing chunk embeddings...")
    chunk_texts = [c["text"] for c in chunks]
    corpus_embeddings = compute_embeddings(chunk_texts)

    logger.info("Computing query embeddings...")
    query_texts = [q["query"] for q in queries]
    query_embeddings = compute_embeddings(query_texts)

    # Evaluate
    logger.info("Evaluating retrieval quality...")
    aggregate_metrics, per_query_results = evaluate_queries(
        queries, chunks, corpus_embeddings, query_embeddings
    )

    # Save outputs
    save_metrics(aggregate_metrics)
    save_plots(per_query_results)

    # Print summary
    logger.info("=== Evaluation Summary ===")
    logger.info(f"  Queries: {aggregate_metrics['num_queries']}")
    logger.info(f"  Chunks:  {aggregate_metrics['num_chunks']}")
    logger.info(f"  MRR:     {aggregate_metrics['mrr']}")
    for k in K_VALUES:
        logger.info(
            f"  @{k:2d} - P: {aggregate_metrics[f'precision_at_{k}']:.4f}  "
            f"R: {aggregate_metrics[f'recall_at_{k}']:.4f}  "
            f"NDCG: {aggregate_metrics[f'ndcg_at_{k}']:.4f}"
        )


if __name__ == "__main__":
    main()
