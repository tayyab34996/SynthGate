"""Expose a small FastAPI interface for contracts and pipeline status."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile

from synthgate.data import deidentify_source, validate_shipments

app = FastAPI(title="SynthGate", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """This endpoint reports whether the API is running."""
    return {"status": "ok"}


@app.get("/contracts")
def list_contracts() -> dict[str, list[str]]:
    """This endpoint lists available scenario contract names."""
    names = sorted(path.stem for path in Path("configs/contracts").glob("*.yaml"))
    return {"contracts": names}


@app.get("/status")
def pipeline_status() -> dict[str, bool]:
    """This endpoint reports whether the core pipeline artifacts exist."""
    return {
        "processed_data": Path("data/processed/train.csv").is_file(),
        "gaussian_copula": Path("models/gaussian_copula.pkl").is_file(),
        "ctgan": Path("models/ctgan.pkl").is_file(),
        "evaluation": Path("artifacts/evaluation/summary.txt").is_file(),
    }


@app.post("/validate")
async def validate_upload(file: UploadFile) -> dict[str, object]:
    """This endpoint validates an uploaded shipment CSV."""
    content = await file.read()
    temporary = Path("artifacts") / "uploaded_dataset.csv"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(content)
    try:
        frame = pd.read_csv(temporary, encoding="latin1")
        clean = validate_shipments(deidentify_source(frame))
        return {"decision": "VALID", "rows": len(clean), "reason": None}
    except (ValueError, pd.errors.ParserError) as error:
        return {"decision": "INVALID", "rows": 0, "reason": str(error)}
