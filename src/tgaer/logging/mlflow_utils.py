from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlflow


def init_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
) -> None:
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_eval_run(
    run_name: str,
    cfg: Dict[str, Any],
    scores: List[float],
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    with mlflow.start_run(run_name=run_name, nested=True):
        flat_params = {k: str(v) for k, v in cfg.items()}
        mlflow.log_params(flat_params)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        metrics = {"avg_score": avg_score}
        if extra_metrics:
            metrics.update(extra_metrics)
        mlflow.log_metrics(metrics)
