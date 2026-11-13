from pathlib import Path

import pandas as pd
import pytest

from synthgate.contracts import evaluate_rules, load_contract
from synthgate.data import SAFE_COLUMNS

CONTRACTS = Path("configs/contracts")


# This test confirms the shipped scenarios preserve the expected stable feature set.
@pytest.mark.parametrize("name", ["normal_operations", "delivery_disruption"])
def test_load_contracts(name: str) -> None:
    contract = load_contract(CONTRACTS / f"{name}.yaml")

    assert contract.name == name
    assert contract.requested_rows == 1000
    assert set(contract.mutable_columns) == {"late_delivery_risk", "delivery_status"}
    assert set(contract.stable_columns).union(contract.mutable_columns) == set(SAFE_COLUMNS)
    assert len(contract.sha256) == 64


# This test rejects contracts whose targets or feature declarations are invalid.
def test_load_contract_rejects_missing_stable_features(tmp_path: Path) -> None:
    raw = (CONTRACTS / "normal_operations.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(raw.replace("  - payment_type\n", ""), encoding="utf-8")

    with pytest.raises(ValueError, match="classify every safe feature"):
        load_contract(path)


# This test reports each invalid row rather than hiding business-rule failures.
def test_evaluate_rules_reports_violations() -> None:
    contract = load_contract(CONTRACTS / "normal_operations.yaml")
    frame = pd.DataFrame(
        {
            "payment_type": ["DEBIT", "CASH"],
            "delivery_status": ["Late delivery", "Shipping on time"],
            "category": ["A", "B"],
            "customer_segment": ["Consumer", "Corporate"],
            "department": ["D", "D"],
            "market": ["M", "M"],
            "order_region": ["R", "R"],
            "order_status": ["COMPLETE", "COMPLETE"],
            "shipping_mode": ["Standard Class", "Same Day"],
            "days_for_shipping": [2, 1],
            "days_for_shipment_scheduled": [2, 1],
            "late_delivery_risk": [1, 0],
            "discount_rate": [0.1, 0.2],
            "quantity": [1, 0],
            "product_price": [10.0, 10.0],
            "sales": [10.0, 10.0],
            "order_item_total": [9.0, 9.0],
            "benefit_per_order": [1.0, 1.0],
            "profit_ratio": [0.1, 0.1],
            "order_to_ship_days": [2.0, 1.0],
        }
    )

    result = evaluate_rules(frame, contract)

    assert result["positive_quantity"] == {"violations": 1, "violation_rate": 0.5}
    assert result["valid_delay_status"] == {"violations": 0, "violation_rate": 0.0}
