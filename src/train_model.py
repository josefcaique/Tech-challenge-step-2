import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = Path("data/raw/online_shoppers_intention.csv")
MODEL_PATH = Path("models/random_forest_model.joblib")
MLFLOW_TRACKING_DIR = Path("mlruns")
MODEL_NAME = "random_forest_revenue_model"
MLFLOW_DB_URI = "sqlite:///mlflow.db"


def main():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_DB_URI)
    os.environ.setdefault("MLFLOW_REGISTRY_URI", MLFLOW_DB_URI)
    shoppers_df = pd.read_csv(DATASET_PATH)
    shoppers_df["Weekend"] = shoppers_df["Weekend"].astype(bool)
    shoppers_df["Revenue"] = shoppers_df["Revenue"].astype(bool)

    categorical_features = ["Month", "VisitorType"]
    numeric_features = [
        c for c in shoppers_df.columns if c not in categorical_features + ["Revenue"]
    ]

    X = shoppers_df.drop(columns=["Revenue"])
    y = shoppers_df["Revenue"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, n_estimators=200, max_depth=8)),
        ]
    )

    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_registry_uri(MLFLOW_DB_URI)

    experiment = mlflow.get_experiment_by_name("tech-challenge-step-2")
    if experiment is None:
        experiment_id = mlflow.create_experiment("tech-challenge-step-2")
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment("tech-challenge-step-2")

    with mlflow.start_run(experiment_id=experiment_id, run_name="random_forest_revenue") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", round(accuracy, 4))
        print(classification_report(y_test, predictions))

        mlflow.set_tags(
            {
                "project": "tech-challenge-step-2",
                "model_type": "random_forest",
                "target": "Revenue",
                "stage": "staging",
            }
        )
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name=MODEL_NAME,
            skops_trusted_types=["numpy.dtype"],
        )

        model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=registered_model.version,
        )
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="staging",
            version=registered_model.version,
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Modelo salvo em", MODEL_PATH)


if __name__ == "__main__":
    main()
