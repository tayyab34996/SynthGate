"""Create release decisions and immutable release manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from synthgate.contracts import load_contract
from synthgate.gates import evaluate_release_gates, release_decision

RELEASE_DIRECTORY: Path = Path("artifacts/releases")


def git_commit() -> str:
    """This function returns the current Git commit hash."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    """This function computes a stable SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_release(
    model_type: str,
    contract_name: str,
    candidate_path: Path,
    evaluation_path: Path,
    sampling_path: Path,
    contract_hash: str,
    mlflow_run_id: str,
) -> dict[str, object]:
    """This function writes the gate decision and a complete release manifest."""
    candidate = pd.read_csv(candidate_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    contract = load_contract(Path("configs/contracts") / f"{contract_name}.yaml")

    evaluation["sampling_status"] = sampling["status"]
    evaluation["maximum_scenario_error"] = contract.max_scenario_error
    evaluation["minimum_stability_score"] = contract.minimum_stability_score
    evaluation["maximum_exact_copy_rate"] = contract.maximum_exact_copy_rate
    gates = evaluate_release_gates(candidate, evaluation)
    decision = release_decision(gates)
    timestamp = datetime.now(UTC).isoformat()
    release_id = (
        f"{timestamp.replace(':', '').replace('+', '_').replace('-', '')}"
        f"-{model_type}-{contract_name}"
    )
    release_dir = RELEASE_DIRECTORY / release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "release_id": release_id,
        "decision": decision["decision"],
        "timestamp": timestamp,
        "git_commit": git_commit(),
        "source_sha256": evaluation["source_sha256"],
        "contract": contract_name,
        "contract_sha256": contract_hash,
        "model_type": model_type,
        "mlflow_run_id": mlflow_run_id,
        "candidate_sha256": file_sha256(candidate_path),
        "evaluation_sha256": file_sha256(evaluation_path),
        "sampling_status": sampling["status"],
        "gate_results": decision["gate_results"],
        "failure_reasons": decision["failure_reasons"],
    }
    (release_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (release_dir / "decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )
    return manifest
