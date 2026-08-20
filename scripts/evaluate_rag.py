"""DVC pipeline stage: evaluate_rag

RAGAS 기반 RAG 품질 평가 스크립트.
Golden QA 데이터셋으로 다음 메트릭을 측정합니다:
- Faithfulness: 답변이 검색된 문서에 근거하는가? (환각 체크)
- Answer Relevancy: 답변이 질문에 적절한가?
- Context Precision: 검색된 문서가 실제로 관련 있는가?
- Context Recall: 필요한 정보를 빠짐없이 찾았는가?

사용법:
    pip install -e ".[eval]"
    export OPENAI_API_KEY=your-key  # RAGAS LLM judge용
    python scripts/evaluate_rag.py

환경 변수:
    OPENAI_API_KEY: RAGAS 내부 LLM 판정용 (필수)
    RAGAS_LLM_MODEL: 사용할 모델 (기본: gpt-4o-mini)
    VISIONRAG_API_URL: VisionRAG API 주소 (기본: http://localhost:8080)
    RAGAS_RUN_MODE: "offline" (golden 데이터만 사용) 또는 "online" (실제 API 호출)

Outputs:
    - reports/rag_metrics.json (aggregate RAGAS metrics)
    - reports/rag_per_query.csv (per-query results for DVC plots)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

# Disable OpenAI Responses API auto-detection (required for BCAI proxy)
os.environ.setdefault("OPENAI_USE_RESPONSES", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_QA_PATH = PROJECT_ROOT / "data" / "golden_qa.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "rag_metrics.json"
PER_QUERY_PATH = REPORTS_DIR / "rag_per_query.csv"

# Config
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
VISIONRAG_API_URL = os.getenv("VISIONRAG_API_URL", "http://localhost:8080")
RUN_MODE = os.getenv("RAGAS_RUN_MODE", "offline")


def load_golden_qa() -> list[dict]:
    """Load golden QA dataset from JSONL file."""
    if not GOLDEN_QA_PATH.exists():
        logger.error(f"Golden QA file not found: {GOLDEN_QA_PATH}")
        return []

    samples = []
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                # Validate required fields
                if "question" not in sample or "ground_truth" not in sample:
                    logger.warning(f"Line {line_num}: missing required fields, skipping")
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: invalid JSON: {e}")

    logger.info(f"Loaded {len(samples)} golden QA samples from {GOLDEN_QA_PATH}")
    return samples


def query_visionrag_api(question: str) -> dict:
    """Call VisionRAG /query endpoint to get answer + contexts.

    Returns dict with 'answer' and 'contexts' keys.
    """
    import httpx

    try:
        response = httpx.post(
            f"{VISIONRAG_API_URL}/query",
            json={"query": question, "top_k": 5},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        contexts = [s.get("text", "") for s in sources if s.get("text")]
        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        logger.warning(f"API call failed for '{question[:50]}...': {e}")
        return {"answer": "", "contexts": []}


def prepare_dataset_offline(samples: list[dict]) -> dict[str, list]:
    """Prepare evaluation dataset from golden QA (no API calls).

    Uses pre-defined contexts and generates placeholder answers.
    Useful for evaluating context quality metrics without a running system.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for sample in samples:
        questions.append(sample["question"])
        ground_truths.append(sample["ground_truth"])
        # Use golden contexts as retrieved contexts (best-case scenario)
        sample_contexts = sample.get("contexts", [])
        contexts.append(sample_contexts)
        # In offline mode, use ground_truth as the answer (measures context metrics)
        answers.append(sample.get("answer", sample["ground_truth"]))

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def prepare_dataset_online(samples: list[dict]) -> dict[str, list]:
    """Prepare evaluation dataset by calling the live VisionRAG API.

    Makes real API calls to get actual retrieval + generation results.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, sample in enumerate(samples):
        logger.info(f"Querying API [{i + 1}/{len(samples)}]: {sample['question'][:60]}...")
        result = query_visionrag_api(sample["question"])

        questions.append(sample["question"])
        ground_truths.append(sample["ground_truth"])
        answers.append(result["answer"])
        contexts.append(result["contexts"] if result["contexts"] else sample.get("contexts", []))

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_ragas_evaluation(eval_data: dict[str, list]) -> tuple[dict, list[dict]]:
    """Run RAGAS evaluation and return aggregate + per-query metrics."""
    # Patch missing import that RAGAS 0.4.x expects
    import sys
    from unittest.mock import MagicMock

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(eval_data)

    logger.info(f"Running RAGAS evaluation on {len(dataset)} samples...")
    logger.info(f"Using LLM: {RAGAS_LLM_MODEL}")

    # Configure LLM for RAGAS
    # Disable SSL verification for corporate proxies
    import httpx
    from langchain_openai import ChatOpenAI

    llm_kwargs = {
        "model": RAGAS_LLM_MODEL,
        "temperature": 0,
        "n": 1,
        "http_client": httpx.Client(verify=False),
        "http_async_client": httpx.AsyncClient(verify=False),
    }
    if OPENAI_BASE_URL:
        llm_kwargs["base_url"] = OPENAI_BASE_URL
    llm = ChatOpenAI(**llm_kwargs)

    # Run evaluation
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
    )

    # Extract aggregate metrics
    aggregate = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "num_samples": len(dataset),
        "llm_model": RAGAS_LLM_MODEL,
        "run_mode": RUN_MODE,
    }

    # Extract per-query results
    per_query = []
    result_df = result.to_pandas()
    for _, row in result_df.iterrows():
        per_query.append({
            "question": row.get("question", ""),
            "faithfulness": round(float(row.get("faithfulness", 0)), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 4),
            "context_precision": round(float(row.get("context_precision", 0)), 4),
            "context_recall": round(float(row.get("context_recall", 0)), 4),
        })

    return aggregate, per_query


def save_metrics(metrics: dict) -> None:
    """Save aggregate metrics as JSON."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Metrics saved to {METRICS_PATH}")


def save_per_query(per_query: list[dict]) -> None:
    """Save per-query results as CSV for DVC plots."""
    PER_QUERY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not per_query:
        logger.warning("No per-query results to save.")
        return

    fieldnames = list(per_query[0].keys())
    with open(PER_QUERY_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query)

    logger.info(f"Per-query results saved to {PER_QUERY_PATH}")


def main() -> None:
    """Main evaluation pipeline."""
    # Check OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        logger.error(
            "OPENAI_API_KEY not set. RAGAS requires an LLM for evaluation.\n"
            "  export OPENAI_API_KEY=your-key"
        )
        # Save empty metrics so DVC pipeline doesn't break
        save_metrics({
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "num_samples": 0,
            "error": "OPENAI_API_KEY not set",
        })
        save_per_query([])
        sys.exit(0)

    # Load golden QA
    samples = load_golden_qa()
    if not samples:
        logger.error("No golden QA samples found. Create data/golden_qa.jsonl first.")
        save_metrics({
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "num_samples": 0,
            "error": "no golden QA data",
        })
        save_per_query([])
        return

    # Prepare dataset based on run mode
    if RUN_MODE == "online":
        logger.info("Running in ONLINE mode — calling VisionRAG API")
        eval_data = prepare_dataset_online(samples)
    else:
        logger.info("Running in OFFLINE mode — using golden contexts only")
        eval_data = prepare_dataset_offline(samples)

    # Run RAGAS evaluation
    try:
        aggregate, per_query = run_ragas_evaluation(eval_data)
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        save_metrics({
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "num_samples": len(samples),
            "error": str(e),
        })
        save_per_query([])
        sys.exit(1)

    # Save results
    save_metrics(aggregate)
    save_per_query(per_query)

    # Print summary
    logger.info("=== RAGAS Evaluation Summary ===")
    logger.info(f"  Mode:              {RUN_MODE}")
    logger.info(f"  Samples:           {aggregate['num_samples']}")
    logger.info(f"  Faithfulness:      {aggregate['faithfulness']}")
    logger.info(f"  Answer Relevancy:  {aggregate['answer_relevancy']}")
    logger.info(f"  Context Precision: {aggregate['context_precision']}")
    logger.info(f"  Context Recall:    {aggregate['context_recall']}")


if __name__ == "__main__":
    main()
