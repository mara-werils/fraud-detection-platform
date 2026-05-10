"""Optuna utilities: study creation, pruning, visualisation, and export.

Usage::

    from ml_pipeline.hyperopt_search import create_study, export_best_params
    study = create_study("xgboost-fraud")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as exc:
    raise ImportError("optuna is required: pip install optuna") from exc

log = structlog.get_logger(__name__)


def create_study(
    study_name: str,
    *,
    direction: str = "maximize",
    storage_path: str | Path | None = None,
    n_startup_trials: int = 10,
    n_warmup_steps: int = 5,
    load_if_exists: bool = True,
) -> optuna.Study:
    """Create (or load) an Optuna study backed by SQLite.

    Args:
        study_name: Human-readable name for the study.
        direction: ``"maximize"`` or ``"minimize"``.
        storage_path: Path for the SQLite database.  Defaults to
            ``optuna_studies/{study_name}.db``.
        n_startup_trials: Random trials before TPE kicks in.
        n_warmup_steps: Steps before pruner can act.
        load_if_exists: If ``True``, resume a previous study.

    Returns:
        A configured ``optuna.Study``.
    """
    if storage_path is None:
        storage_dir = Path("optuna_studies")
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / f"{study_name}.db"

    storage_url = f"sqlite:///{storage_path}"

    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
    )
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )

    log.info(
        "study_created",
        name=study_name,
        direction=direction,
        storage=str(storage_path),
        existing_trials=len(study.trials),
    )
    return study


def export_best_params(
    study: optuna.Study,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export the best trial parameters to a JSON file.

    Args:
        study: A completed Optuna study.
        output_path: Destination JSON path.

    Returns:
        The best parameters dict.
    """
    best = study.best_trial
    params: dict[str, Any] = {
        "trial_number": best.number,
        "value": best.value,
        "params": best.params,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(params, indent=2, default=str))

    log.info("best_params_exported", path=str(output_path), value=best.value)
    return params


def log_study_summary(study: optuna.Study) -> dict[str, Any]:
    """Log and return a summary of the study."""
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
    }
    log.info("study_summary", **summary)
    return summary


def get_param_importance(study: optuna.Study) -> dict[str, float]:
    """Compute hyper-parameter importances (requires completed study)."""
    try:
        importances = optuna.importance.get_param_importances(study)
        log.info("param_importances", importances=importances)
        return dict(importances)
    except Exception:
        log.warning("param_importance_failed", msg="Not enough trials or incompatible evaluator")
        return {}


def save_visualisations(study: optuna.Study, output_dir: str | Path) -> None:
    """Save Optuna visualisation plots as HTML files.

    Requires ``plotly`` to be installed; skips silently otherwise.
    """
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
    except ImportError:
        log.warning("plotly_not_installed", msg="Skipping Optuna visualisations")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plots = {
        "optimization_history": plot_optimization_history,
        "param_importances": plot_param_importances,
        "parallel_coordinate": plot_parallel_coordinate,
        "slice": plot_slice,
    }

    for name, plot_fn in plots.items():
        try:
            fig = plot_fn(study)
            fig.write_html(str(out / f"{name}.html"))
        except Exception:
            log.warning("visualisation_failed", plot=name)

    log.info("visualisations_saved", dir=str(out))
