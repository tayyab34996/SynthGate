import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sdv.sampling import Condition

from synthgate.contracts import load_contract
from synthgate.data import SAFE_COLUMNS
from synthgate.generation import (
    _build_scenario_report,
    _load_synthesizer,
    build_synthesizer,
    generate_scenario,
    set_random_seed,
)
from synthgate.tracking import configure_tracking


class FakeSynthesizer:
    def _set_random_state(self, random_state: int) -> None:
        _ = random_state

    def sample_from_conditions(
        self, conditions: list[Condition], **kwargs: object
    ) -> dict[str, pd.DataFrame]:
        values = conditions[0].column_values
        count = conditions[0].num_rows
        late_flag = values["late_delivery_risk"]
        rows = []
        for _ in range(count):
            rows.append(
                {
                    "payment_type": "DEBIT",
                    "delivery_status": "Late delivery" if late_flag else "Shipping on time",
                    "category": "A",
                    "customer_segment": "Consumer",
                    "department": "D",
                    "market": "M",
                    "order_region": "R",
                    "order_status": "COMPLETE",
                    "shipping_mode": "Standard Class",
                    "days_for_shipping": 2,
                    "days_for_shipment_scheduled": 2,
                    "late_delivery_risk": late_flag,
                    "discount_rate": 0.1,
                    "quantity": 1,
                    "product_price": 10.0,
                    "sales": 10.0,
                    "order_item_total": 9.0,
                    "benefit_per_order": 1.0,
                    "profit_ratio": 0.1,
                    "order_to_ship_days": 2.0,
                }
            )
        return {"shipments": pd.DataFrame(rows)}


# This test checks exact scenario quotas and rule filtering for the sampler.
def test_generate_scenario_returns_requested_delay_rate() -> None:
    contract = replace(
        load_contract(Path("configs/contracts/delivery_disruption.yaml")),
        requested_rows=20,
        batch_size=10,
    )

    generated, report = generate_scenario(FakeSynthesizer(), contract)

    assert len(generated) == 20
    assert generated["late_delivery_risk"].mean() == 0.75
    assert report["status"] == "success"
    assert report["scenario_target_error"] == 0


# This test fails clearly when the generator cannot satisfy mandatory rules.
def test_generate_scenario_reports_bounded_sampling_failure() -> None:
    contract = replace(
        load_contract(Path("configs/contracts/normal_operations.yaml")),
        requested_rows=10,
        batch_size=5,
        max_attempts=2,
    )

    class InvalidSynthesizer(FakeSynthesizer):
        def sample_from_conditions(
            self, conditions: list[Condition], **kwargs: object
        ) -> dict[str, pd.DataFrame]:
            result = super().sample_from_conditions(conditions, **kwargs)["shipments"]
            result.loc[:, "quantity"] = 0
            return {"shipments": result}

    generated, report = generate_scenario(InvalidSynthesizer(), contract)

    assert generated.empty
    assert report["status"] == "unsuccessful"
    assert report["attempts_by_class"]["1"] == 2
    assert "Could not generate" in report["reason"]


# This test keeps run-specific timing and MLflow identifiers out of DVC reports.
def test_scenario_report_contains_only_reproducible_metadata() -> None:
    contract = load_contract(Path("configs/contracts/normal_operations.yaml"))
    sampling_report = {
        "status": "success",
        "reason": None,
        "requested_rows": 1000,
        "generated_rows": 1000,
        "target_delay_proportion": 0.55,
        "achieved_delay_proportion": 0.55,
        "scenario_target_error": 0.0,
        "attempts_by_class": {"0": 450, "1": 550},
        "attempted_rows": 1000,
        "accepted_rows": 1000,
        "acceptance_rate": 1.0,
        "training_seconds": 12.4,
        "generation_seconds": 2.1,
        "run_id": "mlflow-run-id",
    }

    report = _build_scenario_report(sampling_report, "gaussian_copula", contract, 42, "hash")

    assert report["status"] == "success"
    assert report["seed"] == 42
    assert "training_seconds" not in report
    assert "generation_seconds" not in report
    assert "run_id" not in report


# This test verifies that each scenario starts from the saved model's initial RNG state.
def test_scenario_sampling_reloads_a_reproducible_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    row_count = 80
    risk = np.arange(row_count) % 2
    training_data = pd.DataFrame(
        {
            "payment_type": rng.choice(["DEBIT", "TRANSFER"], row_count),
            "delivery_status": np.where(risk, "Late delivery", "Shipping on time"),
            "category": rng.choice(["A", "B", "C"], row_count),
            "customer_segment": rng.choice(["Consumer", "Corporate"], row_count),
            "department": rng.choice(["D1", "D2"], row_count),
            "market": rng.choice(["M1", "M2"], row_count),
            "order_region": rng.choice(["R1", "R2", "R3"], row_count),
            "order_status": rng.choice(["COMPLETE", "PENDING"], row_count),
            "shipping_mode": rng.choice(["Standard", "Express"], row_count),
            "days_for_shipping": rng.integers(1, 11, row_count),
            "days_for_shipment_scheduled": rng.integers(1, 11, row_count),
            "late_delivery_risk": risk,
            "discount_rate": rng.uniform(0, 1, row_count),
            "quantity": rng.integers(1, 6, row_count),
            "product_price": rng.uniform(1, 100, row_count),
            "sales": rng.uniform(1, 500, row_count),
            "order_item_total": rng.uniform(1, 500, row_count),
            "benefit_per_order": rng.uniform(-100, 100, row_count),
            "profit_ratio": rng.uniform(-1, 1, row_count),
            "order_to_ship_days": rng.uniform(1, 10, row_count),
        },
        columns=SAFE_COLUMNS,
    )
    synthesizer, _ = build_synthesizer("gaussian_copula", training_data, {})
    set_random_seed(42)
    synthesizer.fit({"shipments": training_data})
    model_path = tmp_path / "gaussian_copula.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(synthesizer, stream)

    contract = replace(
        load_contract(Path("configs/contracts/delivery_disruption.yaml")),
        requested_rows=12,
        batch_size=6,
    )

    def generate(seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
        set_random_seed(seed)
        scenario_synthesizer = _load_synthesizer(model_path)
        scenario_synthesizer._set_random_state(seed)
        return generate_scenario(scenario_synthesizer, contract)

    first, first_report = generate(42)
    second, second_report = generate(42)
    different_seed, _ = generate(43)

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(different_seed)
    assert first_report["attempted_rows"] == second_report["attempted_rows"]


# This test prevents local MLflow fallbacks when remote credentials are absent.
def test_configure_tracking_requires_remote_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="Remote MLflow credentials are missing"):
        configure_tracking()


# This test forbids accidentally sending experiments to a local MLflow store.
def test_configure_tracking_rejects_local_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "token")

    with pytest.raises(ValueError, match="remote HTTPS server"):
        configure_tracking()
