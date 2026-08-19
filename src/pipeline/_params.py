"""Leitura tipada do arquivo params.yaml usado pelo pipeline DVC."""

from pathlib import Path
from typing import Any

import yaml

PARAMS_FILE = Path("params.yaml")


def load_params(section: str) -> dict[str, Any]:
    """Carrega uma seção específica do params.yaml.

    Args:
        section: Nome da seção (ex.: "preprocess", "train").

    Returns:
        Dicionário com os parâmetros da seção solicitada.

    Raises:
        KeyError: Se a seção não existir no arquivo.
    """
    with PARAMS_FILE.open("r", encoding="utf-8") as file:
        all_params = yaml.safe_load(file)

    if section not in all_params:
        raise KeyError(f"Seção '{section}' não encontrada em {PARAMS_FILE}")

    return all_params[section]
