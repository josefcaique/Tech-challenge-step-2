"""Stage 3 do pipeline DVC: treino do modelo com tracking no MLflow."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.pipeline import model as model_module
from src.pipeline._mlflow import start_run
from src.pipeline._params import load_params

MODEL_NAME = "random_forest_revenue_model"


def main() -> None:
    params = load_params("train")

    target = params["target"]
    features_df = pd.read_parquet(params["features_path"])

    X = features_df.drop(columns=[target])
    y = features_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    categorical_features = [c for c in ("Month", "VisitorType") if c in X.columns]
    pipeline = model_module.build_pipeline(
        features=X_train,
        categorical_features=categorical_features,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )

    with start_run(run_name="train") as run:
        pipeline.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
        test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))

        mlflow.set_tags(
            {
                "project": "tech-challenge-step-2",
                "model_type": "random_forest",
                "target": target,
                "stage": "train",
            }
        )
        mlflow.log_params(
            {
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "test_size": params["test_size"],
                "random_state": params["random_state"],
            }
        )
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_NAME,
            skops_trusted_types=["numpy.dtype"],
        )

        model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_NAME, alias="champion", version=registered_model.version
        )
        client.set_registered_model_alias(
            name=MODEL_NAME, alias="staging", version=registered_model.version
        )

    model_path = Path(params["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_path = Path(params["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}, indent=2)
    )

    print(f"Modelo salvo em {model_path}")
    print(f"Métricas de treino salvas em {metrics_path}")


if __name__ == "__main__":
    main()
