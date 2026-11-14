from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sdv.sampling import Condition

from synthgate.contracts import load_contract
from synthgate.generation import generate_scenario
from synthgate.tracking import configure_tracking


class FakeSynthesizer:
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


# This test prevents local MLflow fallbacks when remote credentials are absent.
def test_configure_tracking_requires_remote_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="Remote MLflow credentials are missing"):
        configure_tracking()
