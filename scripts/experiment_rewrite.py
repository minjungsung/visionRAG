"""검색 실험: rewrite none vs simple 비교.

golden_qa.jsonl의 질문으로 검색하여 정답 문서를 찾는지 비교합니다.
Triton 없이 로컬 임베딩으로 동작합니다.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pymilvus import Collection, connections

from src.config.settings import settings
from src.models.embedding import EmbeddingModel
from src.retrieval.query_rewriter import QueryRewriter

GOLDEN_QA_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_qa.jsonl"
TOP_K = 5


def load_golden_qa():
    samples = []
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def search_milvus(col, embedding, top_k):
    """Milvus에서 직접 검색."""
    results = col.search(
        data=[embedding.tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        output_fields=["text", "doc_id"],
    )
    return [(hit.entity.get("text"), float(hit.score)) for hit in results[0]]


def compute_hit_rate(results, ground_truth_context):
    """검색 결과에 정답 컨텍스트가 포함되어 있는지 체크 (부분 매칭)."""
    for text, score in results:
        # 정답 컨텍스트의 핵심 문구가 검색 결과에 포함되어 있으면 hit
        if ground_truth_context[:50] in text or text[:50] in ground_truth_context:
            return True
    return False


def run_experiment(strategy_name, rewriter, model, col, samples):
    """하나의 전략으로 전체 쿼리 실행."""
    hits = 0
    total = len(samples)
    avg_top_score = 0.0
    total_time = 0.0

    for sample in samples:
        query = sample["question"]
        ground_truth_contexts = sample.get("contexts", [])

        # Rewrite
        t0 = time.perf_counter()
        if strategy_name == "none":
            search_query = query
        elif strategy_name == "simple":
            search_query = rewriter.rewrite(query)
        else:
            search_query = query

        # Embed and search
        embedding = model.encode([search_query])[0]
        results = search_milvus(col, embedding, TOP_K)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Evaluate
        if results:
            avg_top_score += results[0][1]

        for ctx in ground_truth_contexts:
            if compute_hit_rate(results, ctx):
                hits += 1
                break

    hit_rate = hits / total if total > 0 else 0
    avg_score = avg_top_score / total if total > 0 else 0
    avg_latency = (total_time / total) * 1000 if total > 0 else 0

    return {
        "strategy": strategy_name,
        "hit_rate": round(hit_rate, 4),
        "avg_top1_score": round(avg_score, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "hits": hits,
        "total": total,
    }


def main():
    print("=" * 60)
    print("검색 실험: Query Rewrite None vs Simple")
    print("=" * 60)

    # Setup
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    col = Collection(settings.text_collection)
    col.load()

    print("Loading embedding model...")
    model = EmbeddingModel()

    rewriter = QueryRewriter(openai_api_key=None)  # 규칙 기반만

    samples = load_golden_qa()
    print(f"Questions: {len(samples)}")
    print(f"Top-K: {TOP_K}")
    print()

    # Run experiments
    result_none = run_experiment("none", rewriter, model, col, samples)
    result_simple = run_experiment("simple", rewriter, model, col, samples)

    # Print results
    print("-" * 60)
    print(f"{'Metric':<20} {'None':>12} {'Simple':>12} {'Delta':>12}")
    print("-" * 60)
    print(
        f"{'Hit Rate':<20} {result_none['hit_rate']:>12.4f} "
        f"{result_simple['hit_rate']:>12.4f} "
        f"{result_simple['hit_rate'] - result_none['hit_rate']:>+12.4f}"
    )
    print(
        f"{'Avg Top-1 Score':<20} {result_none['avg_top1_score']:>12.4f} "
        f"{result_simple['avg_top1_score']:>12.4f} "
        f"{result_simple['avg_top1_score'] - result_none['avg_top1_score']:>+12.4f}"
    )
    print(
        f"{'Avg Latency (ms)':<20} {result_none['avg_latency_ms']:>12.2f} "
        f"{result_simple['avg_latency_ms']:>12.2f} "
        f"{result_simple['avg_latency_ms'] - result_none['avg_latency_ms']:>+12.2f}"
    )
    print("-" * 60)

    # Save results
    report = {
        "experiment": "rewrite_none_vs_simple",
        "top_k": TOP_K,
        "num_queries": len(samples),
        "results": [result_none, result_simple],
    }

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "experiment_rewrite_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Show rewrite examples
    print("\n--- Rewrite 예시 (상위 3개) ---")
    for sample in samples[:3]:
        q = sample["question"]
        r = rewriter.rewrite(q)
        print(f"  원본: {q}")
        print(f"  변환: {r}")
        print()


if __name__ == "__main__":
    main()
