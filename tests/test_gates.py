import pandas as pd

from synthgate.gates import evaluate_release_gates, release_decision


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "payment_type": ["DEBIT"],
            "delivery_status": ["Shipping on time"],
            "category": ["A"],
            "customer_segment": ["Consumer"],
            "department": ["D"],
            "market": ["M"],
            "order_region": ["R"],
            "order_status": ["COMPLETE"],
            "shipping_mode": ["Standard Class"],
            "days_for_shipping": [2],
            "days_for_shipment_scheduled": [2],
            "late_delivery_risk": [0],
            "discount_rate": [0.1],
            "quantity": [1],
            "product_price": [10.0],
            "sales": [10.0],
            "order_item_total": [9.0],
            "benefit_per_order": [1.0],
            "profit_ratio": [0.1],
            "order_to_ship_days": [2.0],
        }
    )


def _evaluation() -> dict[str, object]:
    return {
        "contract": "normal_operations",
        "scenario_target_error": 0.01,
        "maximum_scenario_error": 0.03,
        "minimum_stability_score": 0.6,
        "maximum_exact_copy_rate": 0.05,
        "sampling_status": "success",
        "rule_results": {
            "no_nulls": {"violations": 0},
            "quantity": {"violations": 0},
        },
        "stability": {"stability_score": 0.7},
        "similarity_risk_indicators": {"training_exact_copy_rate": 0.01},
    }


def test_release_gates_approve_valid_candidate() -> None:
    decision = release_decision(evaluate_release_gates(_frame(), _evaluation()))

    assert decision["decision"] == "APPROVED"
    assert decision["failure_reasons"] == []


def test_release_gates_reject_scenario_and_similarity_defects() -> None:
    evaluation = _evaluation()
    evaluation["scenario_target_error"] = 0.5
    evaluation["similarity_risk_indicators"] = {"training_exact_copy_rate": 0.5}

    decision = release_decision(evaluate_release_gates(_frame(), evaluation))

    assert decision["decision"] == "REJECTED"
    assert any("target error" in reason for reason in decision["failure_reasons"])
    assert any("exact-copy" in reason for reason in decision["failure_reasons"])


def test_release_gates_reject_missing_schema_column() -> None:
    decision = release_decision(
        evaluate_release_gates(_frame().drop(columns=["quantity"]), _evaluation())
    )

    assert decision["decision"] == "REJECTED"
    assert any("Schema" in reason for reason in decision["failure_reasons"])


def test_release_gate_rejects_unsuccessful_sampling() -> None:
    evaluation = _evaluation()
    evaluation["sampling_status"] = "unsuccessful"

    decision = release_decision(evaluate_release_gates(_frame(), evaluation))

    assert decision["decision"] == "REJECTED"
    assert any("Sampling" in reason for reason in decision["failure_reasons"])
