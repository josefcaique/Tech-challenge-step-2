"""Stage 2 do pipeline DVC: engenharia de features."""

from pathlib import Path

import pandas as pd

from src.pipeline._params import load_params


def main() -> None:
    params = load_params("feature_eng")

    input_path = Path(params["input_path"])
    output_path = Path(params["output_path"])
    target = params["target"]
    drop_features = params.get("drop_features", [])

    shoppers_df = pd.read_csv(input_path)

    shoppers_df["TotalPagesViewed"] = (
        shoppers_df["Administrative"] + shoppers_df["Informational"] + shoppers_df["ProductRelated"]
    )
    shoppers_df["TotalDuration"] = (
        shoppers_df["Administrative_Duration"]
        + shoppers_df["Informational_Duration"]
        + shoppers_df["ProductRelated_Duration"]
    )

    shoppers_df = shoppers_df.drop(columns=[c for c in drop_features if c in shoppers_df.columns])
    shoppers_df[target] = shoppers_df[target].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shoppers_df.to_parquet(output_path, index=False)
    print(
        f"Features salvas em {output_path} ({shoppers_df.shape[0]} linhas, {shoppers_df.shape[1]} colunas)"
    )


if __name__ == "__main__":
    main()
