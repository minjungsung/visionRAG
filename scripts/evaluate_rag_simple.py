"""RAG 품질 평가 (LLM-as-judge) — BCAI API 직접 사용.

RAGAS 라이브러리 대신 직접 LLM을 호출하여 평가합니다.
평가 메트릭:
- Faithfulness: 답변이 검색된 문서에 근거하는가?
- Answer Relevancy: 답변이 질문에 적절한가?
- Context Relevancy: 검색된 문서가 질문과 관련 있는가?

사용법:
    PYTHONPATH=. python scripts/evaluate_rag_simple.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_QA_PATH = PROJECT_ROOT / "data" / "golden_qa.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

# BCAI config
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1-codex-mini")


def call_llm(prompt: str, system: str = "") -> str:
    """BCAI API 직접 호출."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        with httpx.Client(verify=False, timeout=60.0) as client:
            response = client.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": MODEL, "messages": messages},
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


def evaluate_faithfulness(answer: str, context: str) -> float:
    """답변이 컨텍스트에 근거하는지 0-1 점수 반환."""
    prompt = f"""다음 답변이 제공된 문서에 근거하여 작성되었는지 평가하세요.

[문서]
{context}

[답변]
{answer}

평가 기준:
- 1.0: 답변의 모든 내용이 문서에 명시적으로 근거함
- 0.5: 일부는 근거하지만 일부는 문서에 없는 내용 포함
- 0.0: 문서와 무관한 답변

점수만 숫자로 출력하세요 (0.0, 0.5, 또는 1.0):"""

    result = call_llm(prompt)
    try:
        return float(result.strip())
    except (ValueError, TypeError):
        return 0.0


def evaluate_relevancy(question: str, answer: str) -> float:
    """답변이 질문에 적절한지 0-1 점수 반환."""
    prompt = f"""다음 답변이 질문에 적절하게 답하고 있는지 평가하세요.

[질문]
{question}

[답변]
{answer}

평가 기준:
- 1.0: 질문에 정확하고 완전하게 답함
- 0.5: 부분적으로 답하거나 불완전함
- 0.0: 질문과 무관한 답변

점수만 숫자로 출력하세요 (0.0, 0.5, 또는 1.0):"""

    result = call_llm(prompt)
    try:
        return float(result.strip())
    except (ValueError, TypeError):
        return 0.0


def evaluate_context_relevancy(question: str, context: str) -> float:
    """검색된 문서가 질문과 관련 있는지 0-1 점수 반환."""
    prompt = f"""다음 문서가 질문에 답하는 데 관련 있는 정보를 포함하고 있는지 평가하세요.

[질문]
{question}

[문서]
{context}

평가 기준:
- 1.0: 문서가 질문에 답하기 위한 핵심 정보를 포함
- 0.5: 부분적으로 관련 있지만 핵심 정보 부족
- 0.0: 질문과 무관한 문서

점수만 숫자로 출력하세요 (0.0, 0.5, 또는 1.0):"""

    result = call_llm(prompt)
    try:
        return float(result.strip())
    except (ValueError, TypeError):
        return 0.0


def load_golden_qa() -> list[dict]:
    samples = []
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    if not API_KEY or not BASE_URL:
        logger.error("OPENAI_API_KEY and OPENAI_BASE_URL must be set.")
        return

    samples = load_golden_qa()
    logger.info(f"Evaluating {len(samples)} samples with LLM-as-judge")
    logger.info(f"Model: {MODEL}")

    # Import RAG pipeline for online evaluation
    from src.retrieval.rag import RAGPipeline

    rag = RAGPipeline()

    results = []
    for i, sample in enumerate(samples):
        question = sample["question"]

        logger.info(f"[{i + 1}/{len(samples)}] {question[:50]}...")

        # Get RAG answer
        t0 = time.perf_counter()
        rag_result = rag.answer(question)
        latency = (time.perf_counter() - t0) * 1000

        answer = rag_result["answer"]
        sources = rag_result["sources"]
        context = "\n".join(s["text"] for s in sources) if sources else ""

        # Evaluate
        faithfulness = evaluate_faithfulness(answer, context)
        relevancy = evaluate_relevancy(question, answer)
        ctx_relevancy = evaluate_context_relevancy(question, context)

        results.append({
            "question": question,
            "answer": answer[:200],
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_relevancy": ctx_relevancy,
            "latency_ms": round(latency, 1),
            "num_sources": len(sources),
        })

        logger.info(
            f"  F={faithfulness:.1f} R={relevancy:.1f} C={ctx_relevancy:.1f} "
            f"latency={latency:.0f}ms"
        )

    # Aggregate
    n = len(results)
    aggregate = {
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4) if n else 0,
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / n, 4) if n else 0,
        "context_relevancy": round(sum(r["context_relevancy"] for r in results) / n, 4) if n else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1) if n else 0,
        "num_samples": n,
        "model": MODEL,
    }

    # Save
    REPORTS_DIR.mkdir(exist_ok=True)
    metrics_path = REPORTS_DIR / "rag_metrics.json"
    metrics_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))

    per_query_path = REPORTS_DIR / "rag_per_query.json"
    per_query_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary
    logger.info("=== RAG Evaluation Summary ===")
    logger.info(f"  Samples:           {aggregate['num_samples']}")
    logger.info(f"  Faithfulness:      {aggregate['faithfulness']}")
    logger.info(f"  Answer Relevancy:  {aggregate['answer_relevancy']}")
    logger.info(f"  Context Relevancy: {aggregate['context_relevancy']}")
    logger.info(f"  Avg Latency:       {aggregate['avg_latency_ms']}ms")
    logger.info(f"  Results saved to:  {metrics_path}")


if __name__ == "__main__":
    main()
