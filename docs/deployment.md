# Deployment and rollback runbook

## Local run

1. Copy `.env.example` to `.env` and add DagsHub credentials.
2. Run `uv sync --locked --extra dev`.
3. Run `uv run dvc repro`.
4. Start the API with `uv run uvicorn api.app:app --host 0.0.0.0 --port 8000`.
5. Start the dashboard with `uv run streamlit run dashboard/app.py`.

## Docker Compose run

1. Run `docker compose -f docker/compose.yaml up --build`.
2. Open `http://localhost:8000/health` for the API.
3. Open `http://localhost:8501` for the dashboard.

## Azure VM

1. Create an Ubuntu VM and allow ports 22, 8000, 8501, and 5000 only where required.
2. Install Docker and Docker Compose.
3. Store credentials in the VM environment or GitHub Actions secrets, never in Git.
4. Clone the repository and run the Compose stack.
5. Confirm DVC can access the DagsHub remote before starting jobs.

## Rollback

1. Stop new promotions.
2. Identify the previous approved model version in MLflow.
3. Call `rollback_model` with that version.
4. Run `uv run dvc repro` and `uv run pytest -q`.
5. Restart the API and dashboard services.
