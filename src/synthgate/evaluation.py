"""Evaluate scenario fulfilment, quality, business rules, and similarity risk."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Final

import pandas as pd
import yaml
from sdmetrics.reports import QualityReport
from sdv.metadata import Metadata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from synthgate.contracts import ScenarioContract, evaluate_rules, load_contract
from synthgate.data import NUMERIC_COLUMNS, SAFE_COLUMNS
from synthgate.tracking import configure_tracking

TABLE_NAME: Final = "shipments"
MODEL_TYPES: Final = ("gaussian_copula", "ctgan")
CONTRACT_DIRECTORY: Final = Path("configs/contracts")


# This function scores only stable columns and explicitly declared stable relationships.
def stability_metrics(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    contract: ScenarioContract,
) -> dict[str, object]:
    reference_stable = reference.loc[:, contract.stable_columns]
    candidate_stable = candidate.loc[:, contract.stable_columns]
    metadata = Metadata.detect_from_dataframes(
        {TABLE_NAME: reference_stable},
        infer_keys=None,
    )
    report = QualityReport()
    report.generate(
        {TABLE_NAME: reference_stable},
        {TABLE_NAME: candidate_stable},
        metadata.to_dict(),
        verbose=False,
    )

    column_details = report.get_details("Column Shapes", table_name=TABLE_NAME)
    column_scores = {
        str(row["Column"]): float(row["Score"])
        for _, row in column_details.iterrows()
        if pd.notna(row["Score"])
    }
    column_shape_score = sum(
        column_scores.get(column, 0.0) for column in contract.stable_columns
    ) / len(contract.stable_columns)

    pair_details = report.get_details("Column Pair Trends", table_name=TABLE_NAME)
    pair_scores = {
        tuple(sorted((str(row["Column 1"]), str(row["Column 2"])))): (
            float(row["Score"]) if pd.notna(row["Score"]) else None
        )
        for _, row in pair_details.iterrows()
    }
    stable_relationship_scores = {
        f"{left}|{right}": pair_scores.get(tuple(sorted((left, right))))
        for left, right in contract.stable_relationships
    }
    relationship_score = sum(
        score if score is not None else 0.0 for score in stable_relationship_scores.values()
    ) / len(contract.stable_relationships)
    return {
        "column_shape_score": column_shape_score,
        "stable_relationship_score": relationship_score,
        "stability_score": statistics.fmean([column_shape_score, relationship_score]),
        "column_scores": column_scores,
        "stable_relationship_scores": stable_relationship_scores,
        "missing_relationships": [
            name for name, score in stable_relationship_scores.items() if score is None
        ],
    }


# This function computes exact-copy and nearest-neighbour risk indicators without PII.
def similarity_metrics(
    training: pd.DataFrame,
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    contract: ScenarioContract,
) -> dict[str, float]:
    numeric_features = [column for column in NUMERIC_COLUMNS if column in contract.stable_columns]
    if not numeric_features:
        raise ValueError("At least one stable numeric feature is required for similarity checks")

    def row_hashes(frame: pd.DataFrame) -> pd.Series:
        return pd.util.hash_pandas_object(frame.loc[:, SAFE_COLUMNS], index=False)

    candidate_hashes = row_hashes(candidate)
    training_hashes = set(row_hashes(training))
    holdout_hashes = set(row_hashes(holdout))
    scaler = MinMaxScaler()
    training_scaled = scaler.fit_transform(training.loc[:, numeric_features])
    holdout_scaled = scaler.transform(holdout.loc[:, numeric_features])
    candidate_scaled = scaler.transform(candidate.loc[:, numeric_features])

    training_neighbors = NearestNeighbors(n_neighbors=1).fit(training_scaled)
    holdout_neighbors = NearestNeighbors(n_neighbors=1).fit(holdout_scaled)
    training_distances = training_neighbors.kneighbors(candidate_scaled, return_distance=True)[0]
    holdout_distances = holdout_neighbors.kneighbors(candidate_scaled, return_distance=True)[0]
    return {
        "training_exact_copy_rate": float(candidate_hashes.isin(training_hashes).mean()),
        "holdout_exact_copy_rate": float(candidate_hashes.isin(holdout_hashes).mean()),
        "nearest_training_distance_mean": float(training_distances.mean()),
        "nearest_training_distance_min": float(training_distances.min()),
        "nearest_holdout_distance_mean": float(holdout_distances.mean()),
        "nearest_holdout_distance_min": float(holdout_distances.min()),
    }


# This function produces all contract-facing metrics for one generated candidate.
def evaluate_candidate(
    training: pd.DataFrame,
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    contract: ScenarioContract,
    sampling_report: dict[str, object],
) -> dict[str, object]:
    if candidate.empty:
        raise ValueError("Cannot evaluate an empty generated dataset")
    missing = sorted(set(SAFE_COLUMNS).difference(candidate.columns))
    if missing:
        raise ValueError(f"Generated dataset is missing features: {', '.join(missing)}")

    achieved = float(candidate["late_delivery_risk"].mean())
    rules = evaluate_rules(candidate, contract)
    stability = stability_metrics(holdout, candidate, contract)
    similarity = similarity_metrics(training, holdout, candidate, contract)
    return {
        "report_version": 1,
        "model_type": sampling_report["model_type"],
        "contract": contract.name,
        "contract_sha256": contract.sha256,
        "source_sha256": sampling_report["source_sha256"],
        "rows": len(candidate),
        "target_delay_proportion": contract.target_delay_proportion,
        "achieved_delay_proportion": achieved,
        "scenario_target_error": abs(achieved - contract.target_delay_proportion),
        "rule_results": rules,
        "stability": stability,
        "similarity_risk_indicators": similarity,
        "operational": {
            "training_seconds": sampling_report["training_seconds"],
            "generation_seconds": sampling_report["generation_seconds"],
            "sampling_attempted_rows": sampling_report["attempted_rows"],
            "sampling_acceptance_rate": sampling_report["acceptance_rate"],
        },
        "privacy_note": (
            "Exact-copy rates and nearest-neighbour distances are risk indicators, "
            "not privacy guarantees."
        ),
    }


# This function writes readable summaries and logs evaluation runs to remote MLflow.
def run_evaluation() -> None:
    import mlflow

    configure_tracking()
    model_config = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    mlflow.set_experiment(str(model_config["experiment_name"]))
    training = pd.read_csv("data/processed/train.csv")
    holdout = pd.read_csv("data/processed/holdout.csv")
    output_dir = Path("artifacts/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[str] = []

    for model_type in MODEL_TYPES:
        for contract_path in sorted(CONTRACT_DIRECTORY.glob("*.yaml")):
            contract = load_contract(contract_path)
            generated_dir = Path("data/generated") / model_type
            candidate = pd.read_csv(generated_dir / f"{contract.name}.csv")
            sampling_report = json.loads(
                (generated_dir / f"{contract.name}.json").read_text(encoding="utf-8")
            )
            evaluation_started = time.perf_counter()
            result = evaluate_candidate(training, holdout, candidate, contract, sampling_report)
            result["evaluation_seconds"] = time.perf_counter() - evaluation_started
            result_path = output_dir / f"{model_type}_{contract.name}.json"
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            stable = result["stability"]
            similarity = result["similarity_risk_indicators"]
            numeric_metrics = {
                "scenario_target_error": float(result["scenario_target_error"]),
                "column_shape_score": float(stable["column_shape_score"]),
                "stable_relationship_score": float(stable["stable_relationship_score"]),
                "stability_score": float(stable["stability_score"]),
                "training_exact_copy_rate": similarity["training_exact_copy_rate"],
                "holdout_exact_copy_rate": similarity["holdout_exact_copy_rate"],
                "nearest_training_distance_mean": similarity["nearest_training_distance_mean"],
                "nearest_holdout_distance_mean": similarity["nearest_holdout_distance_mean"],
                "training_seconds": float(result["operational"]["training_seconds"]),
                "generation_seconds": float(result["operational"]["generation_seconds"]),
                "evaluation_seconds": float(result["evaluation_seconds"]),
            }
            numeric_metrics.update(
                {
                    f"violations_{name}": float(rule["violations"])
                    for name, rule in result["rule_results"].items()
                }
            )
            with mlflow.start_run(run_name=f"evaluation_{model_type}_{contract.name}"):
                mlflow.log_params(
                    {
                        "model_type": model_type,
                        "scenario": contract.name,
                        "contract_sha256": contract.sha256,
                        "source_sha256": sampling_report["source_sha256"],
                        "rows": len(candidate),
                    }
                )
                mlflow.log_metrics(numeric_metrics)
                mlflow.log_artifact(str(result_path), artifact_path="evaluation")

            summaries.append(
                "\n".join(
                    [
                        f"{model_type} / {contract.name}",
                        f"Rows: {len(candidate)}",
                        f"Scenario target error: {result['scenario_target_error']:.4f}",
                        f"Stability score: {stable['stability_score']:.4f}",
                        f"Rule violations: "
                        f"{sum(rule['violations'] for rule in result['rule_results'].values())}",
                        f"Training exact-copy rate: {similarity['training_exact_copy_rate']:.4f}",
                        f"Training / generation seconds: "
                        f"{result['operational']['training_seconds']:.2f} / "
                        f"{result['operational']['generation_seconds']:.2f}",
                    ]
                )
            )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n\n".join(summaries), encoding="utf-8")


# This command-line entry point evaluates every generated model-contract pair.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_evaluation()


if __name__ == "__main__":
    main()
