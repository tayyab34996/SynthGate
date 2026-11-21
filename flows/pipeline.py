"""Prefect tasks for the SynthGate local pipeline."""

from __future__ import annotations

import subprocess

from prefect import flow, task


@task(retries=1)
def run_pipeline_stage(stage: str) -> str:
    """This task runs one DVC stage and returns its combined output."""
    result = subprocess.run(
        ["uv", "run", "dvc", "repro", stage],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout or "Stage completed"


@flow(name="synthgate-pipeline")
def synthgate_pipeline() -> dict[str, str]:
    """This flow runs validation, generation, and evaluation stages."""
    prepare = run_pipeline_stage.submit("prepare_data")
    gaussian = run_pipeline_stage.submit("train_gaussian_copula")
    ctgan = run_pipeline_stage.submit("train_ctgan")
    evaluate = run_pipeline_stage.submit("evaluate")
    return {
        "prepare_data": prepare.result(),
        "train_gaussian_copula": gaussian.result(),
        "train_ctgan": ctgan.result(),
        "evaluate": evaluate.result(),
    }
