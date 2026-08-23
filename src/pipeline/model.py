"""Definição do modelo de classificação (Revenue) usado pelo pipeline DVC."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(
    features: pd.DataFrame,
    categorical_features: list[str],
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> Pipeline:
    """Monta o pipeline sklearn (pré-processamento + RandomForestClassifier).

    Args:
        features: DataFrame de features (sem a coluna alvo) usado para inferir
            quais colunas são numéricas.
        categorical_features: nomes das colunas categóricas a codificar via one-hot.
        n_estimators: número de árvores do RandomForestClassifier.
        max_depth: profundidade máxima das árvores.
        random_state: seed para reprodutibilidade.

    Returns:
        Pipeline sklearn não treinado (preprocessor + classifier).
    """
    numeric_features = [c for c in features.columns if c not in categorical_features]

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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )
