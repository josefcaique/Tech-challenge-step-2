# Tech Challenge Step 2

Pipeline de Machine Learning para prever a **propensão de compra** de usuários de um
e-commerce a partir do comportamento de navegação (dataset [Online Shoppers
Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)),
com dados versionados via **DVC**, pipeline reprodutível em estágios (`dvc.yaml`) e
experimentos rastreados no **MLflow** (Tracking + Model Registry).

## Estrutura do projeto

- [src/pipeline](src/pipeline): estágios do pipeline DVC (`preprocess` → `feature_eng` → `train` → `evaluate`).
- [src/model_registry.py](src/model_registry.py): lógica de promoção de modelo (champion/staging) no MLflow Model Registry.
- [src/train_model.py](src/train_model.py): script de treino standalone (alternativa fora do DVC, usa a mesma lógica de promoção).
- [src/main.py](src/main.py): API FastAPI opcional que serve o modelo `champion` do Registry (`/predict`, `/health`) — não faz parte dos estágios do pipeline nem roda em container, é só um extra de inferência local.
- [notebooks](notebooks): notebooks com análise exploratória.
- [params.yaml](params.yaml): parâmetros de cada estágio do pipeline (paths, hiperparâmetros, split).
- [dvc.yaml](dvc.yaml) / [dvc.lock](dvc.lock): definição e trava do pipeline DVC.
- [data/raw](data/raw): dataset bruto, versionado via DVC (`.dvc` apontando pro remote).
- [data/processed](data/processed): dados intermediários gerados pelos estágios `preprocess`/`feature_eng` (não versionados no git).
- [models](models): artefato do modelo treinado (não versionado no git).
- [metrics](metrics): métricas de treino/avaliação geradas a cada `dvc repro` (versionadas no git, sem cache DVC).
- [mlflow.db](mlflow.db): banco SQLite com o histórico de experimentos do MLflow (gerado localmente).
- [docker](docker): `docker-compose.yml` + Dockerfiles pra rodar o pipeline containerizado (ver seção [Docker](#docker) abaixo).

## Pré-requisitos

- Python 3.10+
- [Poetry](https://python-poetry.org/) 2.x
- [Docker](https://www.docker.com/) + Docker Compose (opcional, só pra rodar via container — ver seção [Docker](#docker))

## Instalação

Instale as dependências do projeto (cria o `.venv` automaticamente):

```bash
poetry install
```

Copie o arquivo de variáveis de ambiente de exemplo (opcional — os valores padrão já
funcionam localmente com SQLite, sem precisar editar nada):

```bash
cp .env.example .env
```

## Como rodar o pipeline (passo a passo)

O projeto usa DVC pra versionar o dataset e orquestrar os estágios do pipeline.

**1. Baixe o dataset versionado no DVC:**

```bash
poetry run dvc pull
```

Isso recupera `data/raw/online_shoppers_intention.csv` a partir do remote local do DVC
(`.dvcstore/`, comitado no próprio repositório — não precisa de credencial nem serviço externo).

**2. Rode o pipeline completo:**

```bash
poetry run dvc repro
```

Isso executa em sequência os 4 estágios definidos em [dvc.yaml](dvc.yaml):

| Estágio | O que faz | Saída |
|---|---|---|
| `preprocess` | limpeza (dedup, cast de tipos) do dataset bruto | `data/processed/online_shoppers_intention_clean.csv` |
| `feature_eng` | features derivadas (`TotalPagesViewed`, `TotalDuration`), drop de colunas | `data/processed/features.parquet` |
| `train` | treino do RandomForest, log no MLflow, promoção champion/staging | `models/random_forest_model.joblib`, `metrics/train_metrics.json` |
| `evaluate` | avaliação no conjunto de teste (accuracy, precision, recall, F1, ROC-AUC) | `metrics/evaluation.json` |

O DVC só reexecuta os estágios cujas dependências (código ou parâmetros em
`params.yaml`) mudaram desde a última run.

**3. (Opcional) Rode só um estágio específico**, por exemplo depois de mudar um
hiperparâmetro em `params.yaml`:

```bash
poetry run dvc repro train
```

**4. (Opcional) Envie o dataset atualizado pro remote**, se você alterou algo em
`data/raw` e fez `dvc add`:

```bash
poetry run dvc push
```

### Alternativa: rodar o treino sem o DVC

```bash
poetry run python src/train_model.py
```

Treina o mesmo tipo de modelo fora da pipeline DVC (lê `data/raw/online_shoppers_intention.csv`
diretamente), salva o artefato em `models/random_forest_model.joblib` e registra o
experimento no MLflow, usando a mesma lógica de promoção champion/staging de
[src/model_registry.py](src/model_registry.py).

## MLflow — visualizando os experimentos

Todo `dvc repro` (estágios `train`/`evaluate`) e toda execução de `train_model.py`
registram um run no MLflow, com parâmetros, métricas e o modelo versionado no Model
Registry sob os aliases `champion` (melhor modelo até agora) e `staging` (última
versão treinada).

Para abrir a UI localmente:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Acesse em [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Docker

O pipeline também roda containerizado, sem precisar instalar Python/Poetry na
máquina host. O `docker/docker-compose.yml` sobe: `postgres` e `minio` (backend
de metadados/artefatos do MLflow), `mlflow` (servidor de tracking) e `ubuntu`
(container com Poetry + o projeto montado, usado pra rodar o pipeline).

**1. Suba os serviços:**

```bash
cd docker
docker compose up -d postgres minio minio-setup mlflow ubuntu
```

**2. Entre no container `ubuntu` e rode o pipeline:**

```bash
docker compose exec ubuntu bash
```

Dentro do container:

```bash
poetry run dvc pull
poetry run dvc repro --force
```

(`--force` é só pra garantir que os estágios rodem mesmo que o `dvc.lock` já
esteja "em dia" com o host — no dia a dia, `dvc repro` sem `--force` já resolve.)

Saia do container com `exit` quando terminar.

**3. Derrube os serviços:**

```bash
docker compose down
```

> **Nota:** o serviço `ubuntu` esconde a pasta `.venv` do host dentro do
> container (via volume anônimo) — isso evita que o Poetry tente reaproveitar
> um ambiente virtual criado no Windows, que não funciona dentro do Linux do
> container.

## Rodando os testes

```bash
poetry run pytest
```

## Resumo do fluxo

```
dvc pull  →  dvc repro  →  (preprocess → feature_eng → train → evaluate)  →  mlflow ui
```

- Dataset bruto versionado com DVC (remote local comitado no repo).
- Pipeline reprodutível em 4 estágios via `dvc.yaml`/`dvc.lock`, local ou via Docker.
- Treino com RandomForest (scikit-learn), tracking completo no MLflow.
- Promoção automática de modelo no Registry apenas quando o F1 supera o champion atual.
