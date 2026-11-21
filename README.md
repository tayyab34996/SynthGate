# SynthGate

SynthGate generates scenario-aware synthetic shipment data and checks each candidate against data contracts before release. It focuses on whether a requested scenario was achieved while preserving the business rules and relationships that should remain stable.

The planned generators are Gaussian Copula (baseline) and CTGAN (main model). The project uses DVC for data/configuration versioning and DagsHub-hosted MLflow for experiment tracking and model registration.

## Requirements

- Git
- Python 3.11 or newer (use the same supported Python version across the team)
- [uv](https://docs.astral.sh/uv/)
- A DagsHub account with this repository connected

## Setup

```powershell
git clone https://github.com/tayyab34996/SynthGate.git
cd SynthGate
uv sync --locked --extra dev
Copy-Item .env.example .env
```

Keep `.env` local. Create a DagsHub access token with repository read/write access, then populate the local `.env` credentials. Never commit tokens, raw datasets, generated datasets, or model weights.

Download the DataCo source file from [Mendeley Data, version 5](https://data.mendeley.net/datasets/8gx2fvg2k6/5) and place `DataCoSupplyChainDataset.csv` under `data/raw/`. Keep the original file local: it includes customer names, email addresses, street details, and password fields. Run `uv run python 1.py` to create a deterministic 10% sample at `data/raw/DataCo_sample.csv`; rerunning the script replaces any existing sample with one generated from the downloaded source. The preparation stage drops direct identifiers, IDs, exact dates, and fine-grained location fields before producing the DVC-managed feature dataset.

### DVC remote

The tracked DVC configuration points to the repository's DagsHub storage remote. Authenticate locally (do not put credentials in `.dvc/config`):

```powershell
dvc remote modify --local storage auth basic
dvc remote modify --local storage user <your-dagshub-username>
dvc remote modify --local storage password <your-dagshub-token>
```

Verify access with `dvc remote list` and `dvc status`. DVC data and model outputs will be stored remotely; Git tracks pipeline output hashes in `dvc.lock`, not the datasets or model weights.

### MLflow tracking

The `.env.example` template provides the DagsHub MLflow tracking URI. Set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` to your DagsHub username and access token in `.env`. Pipeline code requires these settings and logs every training and generation run remotely; it deliberately has no local tracking fallback. The pipeline registers the Gaussian Copula checkpoint and moves its version to Staging.

## Running

Prepare, validate, and split the dataset with the first DVC stage:

```powershell
uv run dvc repro prepare_data
```

This writes a de-identified feature table, stratified train/holdout splits, and a data profile under `data/processed/`. After configuring remote MLflow credentials, train either generator with:

```powershell
uv run dvc repro train_gaussian_copula
uv run dvc repro train_ctgan
```

Each generator trains once, conditionally samples both scenario contracts, applies bounded rejection sampling, and logs training and generation artifacts/metrics to remote MLflow. Evaluate all four model-contract outputs and reproduce every stage with:

```powershell
uv run dvc repro
```

Evaluation reports contain scenario-target error, per-rule violations, SDMetrics scores for stable columns and declared stable relationships, exact-copy rates, and nearest-neighbour distances. Variable training, generation, and evaluation times are logged to MLflow, not stored in DVC-tracked reports, so identical inputs can reproduce the same report contents. Similarity values are risk indicators, not privacy guarantees.

Run `uv run dvc push` to upload DVC-managed outputs to the configured DagsHub remote and `uv run dvc pull` after cloning to restore them. DVC cannot reproduce the data preparation stage if the source CSV has not been downloaded locally. `uv.lock` records the resolved dependency versions; use `uv sync --locked` to reproduce them.

## Release gates

The release engine evaluates schema validity, mandatory rules, scenario target error, stable properties, similarity risk, and sampling success. An approved candidate receives a manifest under `artifacts/releases/`. A rejected candidate keeps its gate failures as an explanation.

## API and dashboard

Start the FastAPI interface:

```powershell
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Start the Streamlit dashboard:

```powershell
uv run streamlit run dashboard/app.py
```

## Docker

Build and start the local stack:

```powershell
docker compose -f docker/compose.yaml up --build
```

## Development

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Contribute on your own named Git branch and commit your own work with your configured Git identity. Each group member should push at least one attributable commit; do not share Git credentials or commit on another member's behalf.

## Repository layout

```text
configs/              Versioned project, ingestion, and scenario configuration
dvc.yaml              Named, reproducible data and model pipeline stages
data/
  raw/                Source CSV (local only; contains identifying source fields)
  processed/          De-identified, validated/split data (DVC-managed)
  generated/          Candidate datasets and sampling reports (DVC-managed)
models/               Generated model weights (local only; DVC-managed)
artifacts/evaluation/ Evaluation reports (local only; DVC-managed)
src/synthgate/        Pipeline package
  data.py             Data preparation and schema validation
  contracts.py        Contract loading and business-rule checks
  generation.py       Gaussian Copula and CTGAN training/sampling
  evaluation.py       SDMetrics, similarity, and operational evaluation
  tracking.py         Remote MLflow setup and registration
tests/                Unit and pipeline tests
.dvc/config           Shared DagsHub DVC remote URL (no credentials)
.env.example          Environment variable template (no credentials)
```

## Data and experiment policy

- Commit code, tests, documentation, configuration, and DVC metadata.
- Do not commit raw/private data, generated datasets, model weights, secrets, or local MLflow stores.
- Use DVC to version meaningful datasets and model artifacts, with the DagsHub remote as their storage.
- Use DagsHub-hosted MLflow for runs and model registration. Credentials must remain in local environment settings.
- Keep any future cloud deployment, containers, REST APIs, and monitoring out of the current scope.
