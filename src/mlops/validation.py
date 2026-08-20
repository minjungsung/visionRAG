"""Deepchecks data & model validation for VisionRAG."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate_embeddings(
    embeddings: np.ndarray,
    labels: list[str] | None = None,
    output_path: str = "reports/embedding_validation.html",
) -> dict:
    """Validate embedding quality using Deepchecks tabular suite."""
    import pandas as pd
    from deepchecks.tabular import Dataset, Suite
    from deepchecks.tabular.checks import (
        FeatureFeatureCorrelation,
        OutlierSampleDetection,
    )

    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
    if labels:
        df["label"] = labels

    dataset = Dataset(df, label="label" if labels else None)

    suite = Suite(
        "Embedding Validation",
        FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
            threshold=0.95, n_pairs=5
        ),
        OutlierSampleDetection(),
    )

    result = suite.run(dataset)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_as_html(output_path)

    return {
        "passed": result.passed(),
        "n_checks": len(result.results),
        "report_path": output_path,
    }


def validate_retrieval_data(
    queries: list[str],
    retrieved_docs: list[list[str]],
    relevance_labels: list[list[int]],
    output_path: str = "reports/retrieval_validation.json",
) -> dict:
    """Validate retrieval quality metrics."""
    results = []
    for query, docs, labels in zip(queries, retrieved_docs, relevance_labels):
        if not labels:
            continue
        # Precision@K
        k = len(labels)
        precision = sum(labels) / k if k > 0 else 0
        # MRR
        mrr = 0.0
        for i, rel in enumerate(labels):
            if rel == 1:
                mrr = 1.0 / (i + 1)
                break
        results.append({"query": query, "precision_at_k": precision, "mrr": mrr})

    avg_precision = np.mean([r["precision_at_k"] for r in results]) if results else 0
    avg_mrr = np.mean([r["mrr"] for r in results]) if results else 0

    report = {
        "avg_precision_at_k": float(avg_precision),
        "avg_mrr": float(avg_mrr),
        "n_queries": len(results),
        "per_query": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def check_data_drift(
    reference_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    output_path: str = "reports/drift_report.html",
) -> dict:
    """Detect embedding distribution drift between reference and current data."""
    import pandas as pd
    from deepchecks.tabular import Dataset, Suite
    from deepchecks.tabular.checks import WholeDatasetDrift

    cols = [f"dim_{i}" for i in range(reference_embeddings.shape[1])]
    ref_df = pd.DataFrame(reference_embeddings, columns=cols)
    cur_df = pd.DataFrame(current_embeddings, columns=cols)

    ref_dataset = Dataset(ref_df)
    cur_dataset = Dataset(cur_df)

    suite = Suite("Drift Detection", WholeDatasetDrift())
    result = suite.run(ref_dataset, cur_dataset)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_as_html(output_path)

    return {"passed": result.passed(), "report_path": output_path}


if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((100, 1024)).astype(np.float32)
    result = validate_embeddings(embeddings)
    print(f"Validation passed: {result['passed']}")
