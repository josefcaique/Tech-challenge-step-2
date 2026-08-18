"""Tests for project configuration files and pipeline metadata."""

from pathlib import Path

import tomllib
import yaml


def _load_pyproject() -> dict:
    with Path("pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_pyproject_has_required_prod_dependencies() -> None:
    """Garante que torch, scikit-learn e mlflow estão nas deps de produção."""
    deps = _load_pyproject()["tool"]["poetry"]["dependencies"]
    for package in ("torch", "scikit-learn", "mlflow"):
        assert package in deps, f"dependência de prod ausente: {package}"


def test_pyproject_has_required_dev_dependencies() -> None:
    """Garante que pytest e ruff estão nas deps de desenvolvimento."""
    dev_deps = _load_pyproject()["tool"]["poetry"]["group"]["dev"]["dependencies"]
    for package in ("pytest", "ruff"):
        assert package in dev_deps, f"dependência de dev ausente: {package}"


def test_pyproject_has_ruff_config() -> None:
    """Garante que a seção [tool.ruff] existe e está configurada."""
    ruff_config = _load_pyproject()["tool"]["ruff"]
    assert "lint" in ruff_config
    assert ruff_config["line-length"] > 0


def test_pre_commit_config_uses_ruff() -> None:
    """Garante que o pre-commit está configurado com o hook do ruff."""
    with Path(".pre-commit-config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    repo_urls = [repo["repo"] for repo in config["repos"]]
    assert any("ruff-pre-commit" in url for url in repo_urls)


def test_dvc_pipeline_has_expected_stages() -> None:
    """Garante que o dvc.yaml define as 4 stages exigidas (>= 3)."""
    with Path("dvc.yaml").open(encoding="utf-8") as file:
        pipeline = yaml.safe_load(file)
    stages = pipeline["stages"]
    assert len(stages) >= 3
    for expected in ("preprocess", "feature_eng", "train", "evaluate"):
        assert expected in stages


def test_dvc_stages_have_deps_params_and_outs() -> None:
    """Garante que cada stage do dvc.yaml declara deps e params (rastreabilidade)."""
    with Path("dvc.yaml").open(encoding="utf-8") as file:
        pipeline = yaml.safe_load(file)
    for name, stage in pipeline["stages"].items():
        assert "cmd" in stage, f"stage sem cmd: {name}"
        assert "deps" in stage, f"stage sem deps: {name}"
        assert "params" in stage, f"stage sem params: {name}"


def test_params_yaml_matches_dvc_stages() -> None:
    """Garante que toda seção referenciada em dvc.yaml existe em params.yaml."""
    with Path("dvc.yaml").open(encoding="utf-8") as file:
        pipeline = yaml.safe_load(file)
    with Path("params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    for stage in pipeline["stages"].values():
        for section in stage["params"]:
            assert section in params, f"seção '{section}' ausente em params.yaml"
