# Tech Challenge Step 2

Este projeto organiza um fluxo completo para treinar um modelo de classificação sobre o dataset de online shoppers intention, salvar o modelo treinado e registrar o experimento no MLflow.

## Estrutura do projeto

- [notebooks](notebooks): notebooks com análise e execução do fluxo.
- [src](src): código Python para treino e preparação do modelo.
- [seeds/online_shoppers_intention.csv](seeds/online_shoppers_intention.csv): dataset base usado para treino.
- [models](models): artefatos gerados pelo treino.
- [mlflow_server.py](mlflow_server.py): script auxiliar que configura o tracking URI do MLflow.
- [mlflow.db](mlflow.db): banco SQLite com o histórico de experimentos do MLflow (gerado localmente).
- [pyproject.toml](pyproject.toml): configuração do projeto e dependências.

## Pré-requisitos

- Python 3.10+
- pip

## Instalação

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação, rode:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Instale as dependências do projeto:

```bash
pip install -e .
```

## Como executar

### Opção 1: rodar o script de treino

```bash
python src/train_model.py
```

Isso treina o modelo, salva o artefato em [models/random_forest_model.joblib](models/random_forest_model.joblib) e registra o experimento no MLflow.

### Opção 2: usar o notebook

1. Abra o notebook em [notebooks](notebooks).
2. Execute as células em ordem.

## MLflow

O treino registra os experimentos em um banco SQLite local ([mlflow.db](mlflow.db)), com os artefatos salvos em `./mlruns`.

Para visualizar os experimentos localmente, com o ambiente virtual ativado, rode:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Acesse no navegador:

```text
http://127.0.0.1:5000
```

## Resultado esperado

O projeto realiza:
- leitura do dataset
- pré-processamento das variáveis
- treino de um modelo de classificação
- avaliação com métricas como acurácia
- registro do experimento no MLflow
