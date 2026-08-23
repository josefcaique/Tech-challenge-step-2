"""Stage 4 do pipeline DVC: avaliação do modelo treinado."""

import json
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.pipeline._mlflow import start_run
from src.pipeline._params import load_params


def main() -> None:
    params = load_params("evaluate")

    target = params["target"]
    features_df = pd.read_parquet(params["features_path"])

    X = features_df.drop(columns=[target])
    y = features_df[target]

    # Mesmo split usado no treino (test_size/random_state), para avaliar sobre
    # o conjunto de teste original sem precisar persistir os índices.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    pipeline = joblib.load(params["model_path"])
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    with start_run(run_name="evaluate"):
        mlflow.set_tags(
            {
                "project": "tech-challenge-step-2",
                "target": target,
                "stage": "evaluate",
            }
        )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            confusion_matrix_path = Path(tmp_dir) / "confusion_matrix.png"
            ConfusionMatrixDisplay.from_predictions(y_test, predictions)
            plt.savefig(confusion_matrix_path)
            plt.close()
            mlflow.log_artifact(str(confusion_matrix_path))

    metrics_path = Path(params["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Métricas de avaliação:", metrics)
    print(f"Métricas salvas em {metrics_path}")


if __name__ == "__main__":
    main()
