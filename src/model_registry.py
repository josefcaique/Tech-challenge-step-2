"""Lógica compartilhada de promoção de modelo no MLflow Model Registry.

Usado tanto pelo stage `train` do pipeline DVC quanto pelo script standalone
`train_model.py`, para não duplicar a decisão de champion/staging.
"""

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "random_forest_revenue_model"
PROMOTION_METRIC = "f1"


def get_champion_metric(client: MlflowClient, model_name: str, metric_name: str) -> float | None:
    """Busca a métrica do modelo atualmente marcado como champion, se existir."""
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None
    return client.get_run(champion.run_id).data.metrics.get(metric_name)


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    version: str,
    metrics: dict[str, float],
    metric_name: str = PROMOTION_METRIC,
) -> bool:
    """Marca a versão como staging sempre, e como champion só se superar a atual."""
    client.set_registered_model_alias(model_name, "staging", version)
    current_best = get_champion_metric(client, model_name, metric_name)
    new_score = metrics[metric_name]
    if current_best is None or new_score > current_best:
        client.set_registered_model_alias(model_name, "champion", version)
        return True
    return False
