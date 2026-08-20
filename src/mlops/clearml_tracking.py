"""ClearML experiment tracking for embedding model training & evaluation."""

from __future__ import annotations

from typing import Any

from src.config.settings import settings


def init_clearml_task(
    task_name: str,
    project_name: str = "VisionRAG",
    task_type: str = "training",
) -> Any:
    """Initialize a ClearML task for experiment tracking."""
    from clearml import Task

    Task.set_credentials(
        api_host=settings.clearml_api_host,
        web_host=settings.clearml_web_host,
        files_host=settings.clearml_files_host,
    )

    task_type_map = {
        "training": Task.TaskTypes.training,
        "testing": Task.TaskTypes.testing,
        "data_processing": Task.TaskTypes.data_processing,
    }

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type_map.get(task_type, Task.TaskTypes.training),
    )
    return task


def log_embedding_experiment(
    task_name: str,
    model_name: str,
    hyperparams: dict,
    metrics: dict,
    dataset_info: dict | None = None,
) -> None:
    """Log an embedding model training or fine-tuning experiment."""
    task = init_clearml_task(task_name, task_type="training")

    task.connect(hyperparams, name="hyperparameters")
    task.connect({"model": model_name}, name="model_config")

    if dataset_info:
        task.connect(dataset_info, name="dataset")

    logger = task.get_logger()
    for key, value in metrics.items():
        logger.report_single_value(name=key, value=value)

    task.close()


def log_retrieval_evaluation(
    eval_name: str,
    metrics: dict[str, float],
    per_query_results: list[dict] | None = None,
) -> None:
    """Log retrieval evaluation results to ClearML."""
    task = init_clearml_task(eval_name, task_type="testing")

    logger = task.get_logger()
    for key, value in metrics.items():
        logger.report_single_value(name=key, value=value)

    if per_query_results:
        import pandas as pd

        df = pd.DataFrame(per_query_results)
        logger.report_table("Per-Query Results", "results", table_plot=df)

    task.close()
