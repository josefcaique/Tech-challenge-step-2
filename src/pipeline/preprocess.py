"""Stage 1 do pipeline DVC: preprocessamento dos dados brutos."""

from pathlib import Path

import pandas as pd

from src.pipeline._params import load_params


def main() -> None:
    params = load_params("preprocess")

    raw_path = Path(params["raw_data_path"])
    processed_path = Path(params["processed_data_path"])

    shoppers_df = pd.read_csv(raw_path)
    shoppers_df = shoppers_df.drop_duplicates()

    shoppers_df["Weekend"] = shoppers_df["Weekend"].astype(bool)
    shoppers_df["Revenue"] = shoppers_df["Revenue"].astype(bool)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    shoppers_df.to_csv(processed_path, index=False)
    print(f"Dataset processado salvo em {processed_path} ({len(shoppers_df)} linhas)")


if __name__ == "__main__":
    main()
