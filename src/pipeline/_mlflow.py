"""Setup compartilhado do MLflow (tracking URI + experimento) para os stages do pipeline."""

import os
from contextlib import contextmanager

import mlflow

MLFLOW_DB_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "tech-challenge-step-2"


@contextmanager
def start_run(run_name: str):
    """Configura o tracking do MLflow e abre um run no experimento do projeto."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_DB_URI)
    os.environ.setdefault("MLFLOW_REGISTRY_URI", MLFLOW_DB_URI)

    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_registry_uri(MLFLOW_DB_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow.create_experiment(EXPERIMENT_NAME)
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        yield run
