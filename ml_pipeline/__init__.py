"""ML training pipeline for the fraud detection platform.

Modules:
    data_prep           – Data loading, splitting, balancing, and scaling.
    feature_engineering – 30+ feature computation organised by group.
    train_xgboost      – XGBoost training with Optuna hyper-parameter search.
    train_gnn          – GraphSAGE GNN training (PyTorch Geometric).
    evaluate           – Model evaluation, SHAP, calibration, latency benchmarks.
    hyperopt_search    – Optuna study helpers and visualisation utilities.
    mlflow_registry    – MLflow experiment tracking and model registry helpers.
"""

__version__ = "0.1.0"
