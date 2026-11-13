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

Download the DataCo source file from [Mendeley Data, version 5](https://data.mendeley.com/datasets/8gx2fvg2k6/5) and place `DataCoSupplyChainDataset.csv` under `data/raw/`. Keep the original file local: it includes customer names, email addresses, street details, and password fields. Run `uv run python 1.py` to create the deterministic sample expected by the pipeline at `data/raw/DataCo_sample.csv`. The preparation stage drops direct identifiers, IDs, exact dates, and fine-grained location fields before producing the DVC-managed feature dataset.

### DVC remote

The tracked DVC configuration points to the repository's DagsHub storage remote. Authenticate locally (do not put credentials in `.dvc/config`):

```powershell
dvc remote modify --local storage auth basic
dvc remote modify --local storage user <your-dagshub-username>
dvc remote modify --local storage password <your-dagshub-token>
```

Verify access with `dvc remote list` and `dvc status`. DVC data and model outputs will be stored remotely; Git should contain the corresponding `.dvc` pointer files, not the data or weights.

### MLflow tracking

The `.env.example` template provides the DagsHub MLflow tracking URI. Set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` to your DagsHub username and access token in `.env`. Pipeline code will load these settings and log runs to the remote tracking server; local MLflow tracking is not the project default.

## Running

Prepare, validate, and split the dataset with the first DVC stage:

```powershell
uv run dvc repro prepare_data
```

This writes a de-identified feature table, stratified train/holdout splits, and a data profile under `data/processed/`. Once the generator and evaluation stages are implemented, run the full pipeline with:

```powershell
uv run dvc repro
```

Run `uv run dvc push` to upload DVC-managed outputs to the configured DagsHub remote and `uv run dvc pull` after cloning to restore them. DVC cannot reconstruct its input if the source CSV has not been downloaded locally. `uv.lock` records the resolved dependency versions; use `uv sync --locked` to reproduce them.

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
  evaluation/         Labeled evaluation data (local only; DVC-managed)
models/               Generated model weights (local only; DVC-managed)
artifacts/            Reports and generated datasets (local only; DVC-managed)
src/synthgate/        Pipeline package
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
