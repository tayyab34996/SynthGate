"""Configure and use the remote MLflow tracking server."""

from __future__ import annotations

import os
from urllib.parse import urlparse


# This function requires HTTPS credentials and prevents accidental local tracking.
def configure_tracking() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    required = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Remote MLflow credentials are missing: {', '.join(missing)}. "
            "Set them in the local .env file."
        )

    uri = os.environ["MLFLOW_TRACKING_URI"]
    parsed = urlparse(uri)
    if parsed.scheme != "https" or parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("MLFLOW_TRACKING_URI must point to a remote HTTPS server")

    import mlflow

    mlflow.set_tracking_uri(uri)
    return uri


# This function registers a checkpoint and explicitly moves its version to Staging.
def register_staging_model(model_uri: str, model_name: str) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    version = mlflow.register_model(model_uri, model_name)
    MlflowClient().transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    return int(version.version)
