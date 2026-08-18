import os
from pathlib import Path
import mlflow

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

TRACKING_URI = "sqlite:///mlflow.db"
Path("mlruns").mkdir(exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

print("MLflow tracking URI:", TRACKING_URI)
print("Run this command in another terminal:")
print("mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")
