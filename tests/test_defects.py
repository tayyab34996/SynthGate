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
        "rule_results": {"no_nulls": {"violations": 0}},
        "stability": {"stability_score": 0.7},
        "similarity_risk_indicators": {"training_exact_copy_rate": 0.01},
    }


def test_negative_quantity_is_detected_by_rules() -> None:
    evaluation = _evaluation()
    evaluation["rule_results"] = {"positive_quantity": {"violations": 1}}

    decision = release_decision(evaluate_release_gates(_frame(), evaluation))

    assert decision["decision"] == "REJECTED"
    assert "1 mandatory rule violations" in decision["failure_reasons"][0]


def test_wrong_target_is_detected_by_scenario_gate() -> None:
    evaluation = _evaluation()
    evaluation["scenario_target_error"] = 0.5

    decision = release_decision(evaluate_release_gates(_frame(), evaluation))

    assert decision["decision"] == "REJECTED"
    assert "target error" in " ".join(decision["failure_reasons"])


def test_collapsed_stability_is_detected() -> None:
    evaluation = _evaluation()
    evaluation["stability"] = {"stability_score": 0.1}

    decision = release_decision(evaluate_release_gates(_frame(), evaluation))

    assert decision["decision"] == "REJECTED"
    assert "Stable properties" in " ".join(decision["failure_reasons"])


def test_copied_rows_are_detected_by_similarity_gate() -> None:
    evaluation = _evaluation()
    evaluation["similarity_risk_indicators"] = {"training_exact_copy_rate": 0.5}

    decision = release_decision(evaluate_release_gates(_frame(), evaluation))

    assert decision["decision"] == "REJECTED"
    assert "exact-copy" in " ".join(decision["failure_reasons"])
