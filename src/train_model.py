"""Treina um classificador de propensão de compra e registra o experimento no MLflow."""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = Path("seeds/online_shoppers_intention.csv")
MODEL_PATH = Path("models/random_forest_model.joblib")
MLFLOW_TRACKING_DIR = Path("mlruns")
MODEL_NAME = "random_forest_revenue_model"
MLFLOW_DB_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "tech-challenge-step-2"
PROMOTION_METRIC = "f1"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def configure_mlflow() -> None:
    """Aponta o MLflow para o tracking store local em SQLite."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_DB_URI)
    os.environ.setdefault("MLFLOW_REGISTRY_URI", MLFLOW_DB_URI)
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_registry_uri(MLFLOW_DB_URI)


def get_or_create_experiment(name: str) -> str:
    """Retorna o id do experimento, criando-o se ainda não existir."""
    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(name)
    mlflow.set_experiment(name)
    return experiment_id


def load_dataset(path: Path) -> pd.DataFrame:
    """Lê o dataset e normaliza as colunas booleanas."""
    shoppers_df = pd.read_csv(path)
    shoppers_df["Weekend"] = shoppers_df["Weekend"].astype(bool)
    shoppers_df["Revenue"] = shoppers_df["Revenue"].astype(bool)
    return shoppers_df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa as colunas em categóricas e numéricas para o pré-processamento."""
    categorical_features = ["Month", "VisitorType"]
    numeric_features = [c for c in df.columns if c not in categorical_features + ["Revenue"]]
    return categorical_features, numeric_features


def build_preprocessor(
    categorical_features: list[str], numeric_features: list[str]
) -> ColumnTransformer:
    """Monta o ColumnTransformer que imputa e escala/codifica as features."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    """Monta o pipeline completo de pré-processamento + RandomForest."""
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, max_depth=8)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def compute_metrics(y_test: pd.Series, predictions, probabilities) -> dict[str, float]:
    """Calcula accuracy, precision, recall, f1 e roc_auc para o conjunto de teste."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary"
    )
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def log_params_and_metrics(model: Pipeline, metrics: dict[str, float]) -> None:
    """Loga tags, hiperparâmetros e métricas do run atual no MLflow."""
    mlflow.set_tags(
        {
            "project": EXPERIMENT_NAME,
            "model_type": "random_forest",
            "target": "Revenue",
        }
    )
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_params(model.named_steps["classifier"].get_params())
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)


def log_model_artifact(model: Pipeline, X_train: pd.DataFrame, report: str) -> None:
    """Loga o classification_report e o modelo treinado (com assinatura) no run atual."""
    mlflow.log_text(report, "classification_report.txt")
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        name=MODEL_NAME,
        signature=signature,
        input_example=X_train.head(3),
        skops_trusted_types=["numpy.dtype"],
    )


def register_model_version(run_id: str):
    """Registra a versão do modelo logado no run informado."""
    model_uri = f"runs:/{run_id}/{MODEL_NAME}"
    return mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)


def get_champion_metric(client: MlflowClient, metric_name: str) -> float | None:
    """Busca a métrica do modelo atualmente marcado como champion, se existir."""
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None
    return client.get_run(champion.run_id).data.metrics.get(metric_name)


def promote_if_better(client: MlflowClient, version: str, metrics: dict[str, float]) -> bool:
    """Marca a versão como staging sempre, e como champion só se superar a atual."""
    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    current_best = get_champion_metric(client, PROMOTION_METRIC)
    new_score = metrics[PROMOTION_METRIC]
    if current_best is None or new_score > current_best:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        return True
    return False


def save_local_copy(model: Pipeline, path: Path) -> None:
    """Salva uma cópia local do modelo treinado em disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def train_and_evaluate(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float], str]:
    """Treina o pipeline e calcula métricas + classification_report no conjunto de teste."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, predictions, probabilities)
    report = classification_report(y_test, predictions)
    return metrics, report


def promote_and_report(version: str, metrics: dict[str, float]) -> None:
    """Decide a promoção do modelo no Registry e imprime o resultado."""
    client = MlflowClient()
    promoted = promote_if_better(client, version, metrics)
    status = "promoted to champion" if promoted else "kept as staging (did not beat champion)"
    print(f"Model version {version} {status} - {PROMOTION_METRIC}={metrics[PROMOTION_METRIC]:.4f}")


def prepare_data(path: Path) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carrega o dataset, faz o split e monta o pipeline pronto para treino."""
    df = load_dataset(path)
    categorical_features, numeric_features = get_feature_columns(df)
    X, y = df.drop(columns=["Revenue"]), df["Revenue"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = build_pipeline(categorical_features, numeric_features)
    return model, X_train, X_test, y_train, y_test


def main() -> None:
    """Treina, avalia, rastreia no MLflow e promove o modelo no Registry."""
    configure_mlflow()
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    model, X_train, X_test, y_train, y_test = prepare_data(DATASET_PATH)

    with mlflow.start_run(experiment_id=experiment_id, run_name="random_forest_revenue") as run:
        metrics, report = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        print(f"Accuracy: {metrics['accuracy']:.4f}\n{report}")
        log_params_and_metrics(model, metrics)
        log_model_artifact(model, X_train, report)
        registered_model = register_model_version(run.info.run_id)
        promote_and_report(registered_model.version, metrics)

    save_local_copy(model, MODEL_PATH)
    print("Modelo salvo em", MODEL_PATH)


if __name__ == "__main__":
    main()
