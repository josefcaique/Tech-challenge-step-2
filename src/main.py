"""API FastAPI que serve o modelo `random_forest_revenue_model` a partir do
MLflow Model Registry (alias `champion`), para inferência de propensão de
compra (Revenue) sobre sessões de e-commerce.

Rodar localmente:
    poetry run uvicorn src.main:app --reload

Extra opcional do projeto — não faz parte do pipeline DVC nem roda em
container; serve só pra testar inferência local a partir do modelo já
registrado no MLflow Registry.
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = "random_forest_revenue_model"
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

_model_state: dict = {"model": None, "version": None}


class SessionFeatures(BaseModel):
    """Comportamento de navegação de uma sessão, no mesmo formato do dataset bruto."""

    Administrative: int = Field(..., ge=0, description="Nº de páginas administrativas visitadas")
    Administrative_Duration: float = Field(
        ..., ge=0, description="Tempo total em páginas administrativas (s)"
    )
    Informational: int = Field(..., ge=0, description="Nº de páginas informativas visitadas")
    Informational_Duration: float = Field(
        ..., ge=0, description="Tempo total em páginas informativas (s)"
    )
    ProductRelated: int = Field(..., ge=0, description="Nº de páginas de produto visitadas")
    ProductRelated_Duration: float = Field(
        ..., ge=0, description="Tempo total em páginas de produto (s)"
    )
    BounceRates: float = Field(..., ge=0, le=1)
    ExitRates: float = Field(..., ge=0, le=1)
    PageValues: float = Field(..., ge=0)
    SpecialDay: float = Field(
        ..., ge=0, le=1, description="Proximidade a uma data especial (0 a 1)"
    )
    Month: str = Field(..., examples=["Feb", "Nov", "Dec"])
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str = Field(..., examples=["Returning_Visitor", "New_Visitor", "Other"])

    def to_model_features(self) -> pd.DataFrame:
        """Reproduz a engenharia de features de src/pipeline/feature_eng.py
        (TotalPagesViewed/TotalDuration) para uma única sessão."""
        row = self.model_dump()
        row["TotalPagesViewed"] = self.Administrative + self.Informational + self.ProductRelated
        row["TotalDuration"] = (
            self.Administrative_Duration
            + self.Informational_Duration
            + self.ProductRelated_Duration
        )
        return pd.DataFrame([row])


class PredictionResponse(BaseModel):
    will_purchase: bool
    purchase_probability: float
    model_alias: str
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "model_unavailable"]
    model_name: str
    model_alias: str
    model_version: str | None = None


def _load_champion_model() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    _model_state["model"] = mlflow.sklearn.load_model(model_uri)
    _model_state["version"] = version.version


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _load_champion_model()
    except Exception as exc:  # noqa: BLE001 - queremos subir a API mesmo sem modelo disponível
        print(f"[startup] não foi possível carregar o modelo '{MODEL_NAME}@{MODEL_ALIAS}': {exc}")
    yield


app = FastAPI(
    title="Purchase Propensity API",
    description="Prevê a propensão de compra de uma sessão de e-commerce.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _model_state["model"] is None:
        return HealthResponse(
            status="model_unavailable", model_name=MODEL_NAME, model_alias=MODEL_ALIAS
        )
    return HealthResponse(
        status="ok",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version=str(_model_state["version"]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(session: SessionFeatures) -> PredictionResponse:
    if _model_state["model"] is None:
        try:
            _load_champion_model()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo '{MODEL_NAME}@{MODEL_ALIAS}' indisponível no MLflow Registry: {exc}",
            ) from exc

    features = session.to_model_features()
    probability = float(_model_state["model"].predict_proba(features)[0][1])

    return PredictionResponse(
        will_purchase=probability >= 0.5,
        purchase_probability=probability,
        model_alias=MODEL_ALIAS,
        model_version=str(_model_state["version"]),
    )
