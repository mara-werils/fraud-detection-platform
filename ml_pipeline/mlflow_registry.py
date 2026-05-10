"""MLflow integration: experiment tracking, model registry, and helpers.

Usage::

    from ml_pipeline.mlflow_registry import init_experiment, log_model
    init_experiment("fraud-xgboost")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError as exc:
    raise ImportError("mlflow is required: pip install mlflow") from exc

log = structlog.get_logger(__name__)

_client: MlflowClient | None = None


def _get_client() -> MlflowClient:
    global _client
    if _client is None:
        _client = MlflowClient()
    return _client


# ---------------------------------------------------------------------------
# Experiment management
# ---------------------------------------------------------------------------

def init_experiment(
    experiment_name: str,
    *,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> str:
    """Create or get an MLflow experiment and set it as active.

    Args:
        experiment_name: Human-readable experiment name.
        tracking_uri: MLflow tracking server URI.  Falls back to
            ``MLFLOW_TRACKING_URI`` env var, then ``./mlruns``.
        artifact_location: Root artifact store location.

    Returns:
        The experiment ID.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
        log.info("experiment_created", name=experiment_name, id=exp_id)
    else:
        exp_id = experiment.experiment_id
        log.info("experiment_exists", name=experiment_name, id=exp_id)

    mlflow.set_experiment(experiment_name)
    return exp_id


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def start_run(
    run_name: str | None = None,
    *,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> mlflow.ActiveRun:
    """Start an MLflow run and return the ActiveRun context manager.

    Args:
        run_name: Optional human-readable run name.
        tags: Extra tags to attach.
        nested: Whether this is a nested (child) run.

    Returns:
        ``mlflow.ActiveRun`` (use as context manager).
    """
    return mlflow.start_run(run_name=run_name, tags=tags, nested=nested)


def log_params(params: dict[str, Any]) -> None:
    """Log a dictionary of parameters to the active run."""
    # MLflow params must be strings of <=500 chars
    sanitised = {k: str(v)[:500] for k, v in params.items()}
    mlflow.log_params(sanitised)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of metrics to the active run."""
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str | Path, artifact_subdir: str | None = None) -> None:
    """Log a local file as an artifact."""
    mlflow.log_artifact(str(local_path), artifact_path=artifact_subdir)


def log_model(
    model: Any,
    artifact_path: str,
    *,
    model_type: str = "xgboost",
    registered_name: str | None = None,
    input_example: Any = None,
) -> str:
    """Log a trained model to the active MLflow run.

    Args:
        model: The trained model object.
        artifact_path: Sub-path within the run artifacts.
        model_type: One of ``"xgboost"``, ``"sklearn"``, ``"pytorch"``.
        registered_name: If set, also register the model in the registry.
        input_example: Optional input example for signature inference.

    Returns:
        The model URI (``runs:/<run_id>/<artifact_path>``).
    """
    log_fn_map = {
        "xgboost": mlflow.xgboost.log_model,
        "sklearn": mlflow.sklearn.log_model,
    }

    if model_type in log_fn_map:
        info = log_fn_map[model_type](
            model,
            artifact_path,
            registered_model_name=registered_name,
            input_example=input_example,
        )
    elif model_type == "pytorch":
        info = mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_name,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model_uri = info.model_uri
    log.info("model_logged", uri=model_uri, registered_name=registered_name)
    return model_uri


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def transition_model_stage(
    model_name: str,
    version: int | str,
    stage: str,
    *,
    archive_existing: bool = True,
) -> None:
    """Transition a registered model version to a new stage.

    Args:
        model_name: Registered model name.
        version: Model version number.
        stage: Target stage (``"Staging"``, ``"Production"``, ``"Archived"``).
        archive_existing: If ``True``, archive the current model in that stage.
    """
    client = _get_client()

    if archive_existing:
        # Archive any existing model in the target stage
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == stage:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                )

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
    )
    log.info("model_stage_transitioned", name=model_name, version=version, stage=stage)


def load_production_model(model_name: str, model_type: str = "xgboost") -> Any:
    """Load the latest Production model from the registry.

    Args:
        model_name: Registered model name.
        model_type: One of ``"xgboost"``, ``"sklearn"``, ``"pytorch"``.

    Returns:
        The loaded model object.
    """
    model_uri = f"models:/{model_name}/Production"

    load_fn_map = {
        "xgboost": mlflow.xgboost.load_model,
        "sklearn": mlflow.sklearn.load_model,
        "pytorch": mlflow.pytorch.load_model,
    }

    load_fn = load_fn_map.get(model_type)
    if load_fn is None:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = load_fn(model_uri)
    log.info("production_model_loaded", name=model_name, uri=model_uri)
    return model


def get_latest_version(model_name: str, stage: str = "Production") -> int | None:
    """Return the latest version number for a model in the given stage."""
    client = _get_client()
    versions = client.search_model_versions(f"name='{model_name}'")
    for mv in sorted(versions, key=lambda x: int(x.version), reverse=True):
        if mv.current_stage == stage:
            return int(mv.version)
    return None
