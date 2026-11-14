"""Train the baseline or main synthesizer and generate contract-aware datasets."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import pandas as pd
import torch
import yaml
from sdv.metadata import Metadata
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

from synthgate.contracts import ScenarioContract, load_contract, valid_rows_mask
from synthgate.tracking import configure_tracking, register_staging_model

TABLE_NAME: Final = "shipments"
CONTRACT_DIRECTORY: Final = Path("configs/contracts")


class SynthesizerProtocol(Protocol):
    def fit(self, data: dict[str, pd.DataFrame]) -> None: ...

    def sample(self, num_rows: int) -> dict[str, pd.DataFrame]: ...

    def sample_from_conditions(
        self, conditions: list[Condition], **kwargs: object
    ) -> dict[str, pd.DataFrame]: ...


# This function seeds Python, NumPy, and PyTorch for repeatable CPU training and sampling.
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# This function creates a synthesizer with the contract's binary target as a category.
def build_synthesizer(
    model_type: str, training_data: pd.DataFrame, model_config: Mapping[str, object]
) -> tuple[SynthesizerProtocol, Metadata]:
    metadata = Metadata.detect_from_dataframes(
        {TABLE_NAME: training_data},
        infer_keys=None,
    )
    metadata.update_column(
        "late_delivery_risk",
        table_name=TABLE_NAME,
        sdtype="categorical",
    )

    if model_type == "gaussian_copula":
        synthesizer = GaussianCopulaSynthesizer(metadata)
    elif model_type == "ctgan":
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=int(model_config["epochs"]),
            embedding_dim=int(model_config["embedding_dim"]),
            generator_dim=tuple(model_config["generator_dim"]),
            discriminator_dim=tuple(model_config["discriminator_dim"]),
            batch_size=int(model_config["batch_size"]),
            pac=int(model_config["pac"]),
            enable_gpu=False,
        )
    else:
        raise ValueError(f"Unknown synthesizer: {model_type}")
    return synthesizer, metadata


# This function generates exact class quotas and rejects rows that violate the contract.
def generate_scenario(
    synthesizer: SynthesizerProtocol,
    contract: ScenarioContract,
    table_name: str = TABLE_NAME,
) -> tuple[pd.DataFrame, dict[str, object]]:
    requested_rows = contract.requested_rows
    requested_late = round(requested_rows * contract.target_delay_proportion)
    quotas = {1: requested_late, 0: requested_rows - requested_late}
    accepted: list[pd.DataFrame] = []
    class_attempts: dict[str, int] = {}
    sampled_rows = 0
    accepted_rows = 0
    reason: str | None = None

    for late_flag, quota in quotas.items():
        class_attempts[str(late_flag)] = 0
        class_rows: list[pd.DataFrame] = []
        class_accepted = 0
        while class_accepted < quota and class_attempts[str(late_flag)] < contract.max_attempts:
            class_attempts[str(late_flag)] += 1
            needed = quota - class_accepted
            batch_rows = min(needed, contract.batch_size)
            conditions = {"late_delivery_risk": late_flag}
            if late_flag == 1:
                conditions["delivery_status"] = "Late delivery"
            sampled = synthesizer.sample_from_conditions(
                [
                    Condition(
                        conditions,
                        num_rows=batch_rows,
                        table_name=table_name,
                    )
                ]
            )[table_name]
            sampled_rows += len(sampled)
            valid = valid_rows_mask(sampled, contract)
            valid &= sampled["late_delivery_risk"].eq(late_flag)
            valid_batch = sampled.loc[valid].head(needed)
            class_rows.append(valid_batch)
            class_accepted += len(valid_batch)
            accepted_rows += len(valid_batch)

        if class_rows:
            accepted.append(pd.concat(class_rows, ignore_index=True))
        if class_accepted < quota:
            reason = (
                f"Could not generate {quota} valid rows for late_delivery_risk={late_flag}; "
                f"accepted {class_accepted} after {class_attempts[str(late_flag)]} attempts"
            )
            break

    generated = (
        pd.concat(accepted, ignore_index=True)
        if accepted
        else pd.DataFrame(columns=contract.stable_columns + contract.mutable_columns)
    )
    achieved = float(generated["late_delivery_risk"].mean()) if len(generated) else 0.0
    report: dict[str, object] = {
        "status": "success" if len(generated) == requested_rows else "unsuccessful",
        "reason": reason,
        "requested_rows": requested_rows,
        "generated_rows": len(generated),
        "target_delay_proportion": contract.target_delay_proportion,
        "achieved_delay_proportion": achieved,
        "scenario_target_error": abs(achieved - contract.target_delay_proportion),
        "attempts_by_class": class_attempts,
        "attempted_rows": sampled_rows,
        "accepted_rows": accepted_rows,
        "acceptance_rate": accepted_rows / sampled_rows if sampled_rows else 0.0,
    }
    return generated, report


# This function serializes the fitted generator as a usable MLflow PyFunc checkpoint.
def _log_pyfunc_model(synthesizer: SynthesizerProtocol) -> str:
    import mlflow

    class SynthGatePyFunc(mlflow.pyfunc.PythonModel):
        def __init__(self, fitted_synthesizer: SynthesizerProtocol) -> None:
            self.fitted_synthesizer = fitted_synthesizer

        def predict(
            self,
            context: object,
            model_input: pd.DataFrame,
            params: dict[str, object] | None = None,
        ) -> pd.DataFrame:
            _ = context, params
            if "num_rows" not in model_input:
                raise ValueError("Input must include a positive num_rows value")
            counts = model_input["num_rows"].astype(int)
            if counts.le(0).any():
                raise ValueError("num_rows values must be positive")
            frames = [self.fitted_synthesizer.sample(int(count))[TABLE_NAME] for count in counts]
            return pd.concat(frames, ignore_index=True)

    model_info = mlflow.pyfunc.log_model(
        name="generator",
        python_model=SynthGatePyFunc(synthesizer),
        input_example=pd.DataFrame({"num_rows": [5]}),
        code_paths=[str(Path(__file__).resolve().parents[1])],
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"pandas=={pd.__version__}",
            f"sdv=={__import__('sdv').__version__}",
        ],
    )
    return model_info.model_uri


# This function trains one generator, logs each run remotely, and writes DVC outputs.
def run_generation(model_type: str, config_path: Path = Path("configs/models.yaml")) -> None:
    import mlflow

    tracking_uri = configure_tracking()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["models"][model_type]
    seed = int(config["seed"])
    table_name = str(config["table_name"])
    training_data = pd.read_csv("data/processed/train.csv")
    training_classes = set(training_data["late_delivery_risk"].unique())
    if training_classes != {0, 1}:
        raise ValueError("Training data must contain both delivery-risk classes")

    set_random_seed(seed)
    synthesizer, metadata = build_synthesizer(model_type, training_data, model_config)
    mlflow.set_experiment(str(config["experiment_name"]))
    model_path = Path("models") / f"{model_type}.pkl"
    metadata_path = Path("models") / f"{model_type}_metadata.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = str(json.loads(Path("data/processed/profile.json").read_text())["source_sha256"])

    with mlflow.start_run(run_name=f"{model_type}_training") as training_run:
        mlflow.log_params(
            {
                "model_type": model_type,
                "seed": seed,
                "training_rows": len(training_data),
                "source_sha256": source_hash,
                "tracking_uri": tracking_uri,
                **{f"model_{key}": value for key, value in model_config.items()},
            }
        )
        training_started = time.perf_counter()
        synthesizer.fit({table_name: training_data})
        training_seconds = time.perf_counter() - training_started
        with model_path.open("wb") as stream:
            pickle.dump(synthesizer, stream)
        metadata.save_to_json(str(metadata_path))
        mlflow.log_metric("training_seconds", training_seconds)
        mlflow.log_artifact(str(model_path), artifact_path="checkpoint")
        mlflow.log_artifact(str(metadata_path), artifact_path="checkpoint")
        model_uri = _log_pyfunc_model(synthesizer)
        if bool(model_config["register"]):
            register_staging_model(model_uri, str(model_config["registry_name"]))

    output_dir = Path("data/generated") / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ids: dict[str, object] = {
        "training": training_run.info.run_id,
        "training_seconds": training_seconds,
        "scenarios": {},
    }
    for contract_path in sorted(CONTRACT_DIRECTORY.glob("*.yaml")):
        contract = load_contract(contract_path)
        set_random_seed(seed)
        generation_started = time.perf_counter()
        generated, report = generate_scenario(synthesizer, contract, table_name)
        report["model_type"] = model_type
        report["contract"] = contract.name
        report["contract_sha256"] = contract.sha256
        report["seed"] = seed
        report["source_sha256"] = source_hash
        report["training_seconds"] = training_seconds
        report["generation_seconds"] = time.perf_counter() - generation_started

        dataset_path = output_dir / f"{contract.name}.csv"
        report_path = output_dir / f"{contract.name}.json"
        generated.to_csv(dataset_path, index=False)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        with mlflow.start_run(run_name=f"{model_type}_{contract.name}") as scenario_run:
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "scenario": contract.name,
                    "contract_sha256": contract.sha256,
                    "seed": seed,
                    "target_delay_proportion": contract.target_delay_proportion,
                    "requested_rows": contract.requested_rows,
                }
            )
            mlflow.log_metrics(
                {
                    "generated_rows": float(report["generated_rows"]),
                    "achieved_delay_proportion": float(report["achieved_delay_proportion"]),
                    "scenario_target_error": float(report["scenario_target_error"]),
                    "generation_seconds": float(report["generation_seconds"]),
                    "acceptance_rate": float(report["acceptance_rate"]),
                    "attempted_rows": float(report["attempted_rows"]),
                }
            )
            mlflow.log_artifact(str(dataset_path), artifact_path="generated")
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            run_ids["scenarios"][contract.name] = scenario_run.info.run_id
            if report["status"] != "success":
                raise RuntimeError(str(report["reason"]))

    (output_dir / "runs.json").write_text(json.dumps(run_ids, indent=2), encoding="utf-8")


# This command-line entry point runs the configured baseline or main generator.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gaussian_copula", "ctgan"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    args = parser.parse_args()
    run_generation(args.model, args.config)


if __name__ == "__main__":
    main()
