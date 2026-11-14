from dataclasses import replace
from pathlib import Path

import pandas as pd

from synthgate.contracts import load_contract
from synthgate.evaluation import evaluate_candidate


def _build_frame(rows: int = 100) -> pd.DataFrame:
    records = []
    for index in range(rows):
        risk = int(index % 2 == 0)
        quantity = index % 5 + 1
        product_price = quantity * 10.0
        segment = "Consumer" if index % 2 else "Corporate"
        records.append(
            {
                "payment_type": ("DEBIT", "CASH")[index % 2],
                "delivery_status": "Late delivery" if risk else "Shipping on time",
                "category": f"Category {index % 4}",
                "customer_segment": segment,
                "department": "D",
                "market": "USCA" if segment == "Consumer" else "Europe",
                "order_region": f"Region {index % 3}",
                "order_status": "COMPLETE",
                "shipping_mode": "Standard Class" if index % 2 else "First Class",
                "days_for_shipping": index % 5 + 1,
                "days_for_shipment_scheduled": 4 if index % 2 else 2,
                "late_delivery_risk": risk,
                "discount_rate": (index % 5) / 100,
                "quantity": quantity,
                "product_price": product_price,
                "sales": product_price,
                "order_item_total": product_price - 1,
                "benefit_per_order": quantity * 2.0,
                "profit_ratio": 0.1,
                "order_to_ship_days": index % 5 + 1.0,
            }
        )
    return pd.DataFrame.from_records(records)


# This test confirms the evaluator measures stable fields and preserves risk indicators.
def test_evaluate_candidate_scores_real_holdout_and_exact_copies() -> None:
    frame = _build_frame()
    contract = replace(
        load_contract(Path("configs/contracts/normal_operations.yaml")),
        target_delay_proportion=0.5,
    )
    sampling = {
        "model_type": "gaussian_copula",
        "source_sha256": "source-hash",
        "training_seconds": 2.0,
        "generation_seconds": 1.0,
        "attempted_rows": len(frame),
        "acceptance_rate": 1.0,
    }

    result = evaluate_candidate(frame, frame, frame.copy(), contract, sampling)

    assert result["scenario_target_error"] == 0
    assert result["stability"]["stability_score"] == 1
    assert len(result["stability"]["stable_relationship_scores"]) == len(
        contract.stable_relationships
    )
    assert result["similarity_risk_indicators"]["training_exact_copy_rate"] == 1
    assert "not privacy guarantees" in result["privacy_note"]


# This test ensures broken stable relationships score below an unchanged holdout.
def test_independent_column_shuffles_reduce_stability_score() -> None:
    frame = _build_frame()
    contract = replace(
        load_contract(Path("configs/contracts/normal_operations.yaml")),
        target_delay_proportion=0.5,
    )
    shuffled = frame.copy()
    stable_columns = contract.stable_columns
    for position, column in enumerate(stable_columns):
        shuffled[column] = shuffled[column].sample(frac=1, random_state=position).to_numpy()
    sampling = {
        "model_type": "gaussian_copula",
        "source_sha256": "source-hash",
        "training_seconds": 2.0,
        "generation_seconds": 1.0,
        "attempted_rows": len(frame),
        "acceptance_rate": 1.0,
    }

    real_result = evaluate_candidate(frame, frame, frame.copy(), contract, sampling)
    shuffled_result = evaluate_candidate(frame, frame, shuffled, contract, sampling)

    assert real_result["stability"]["stability_score"] == 1
    assert (
        shuffled_result["stability"]["stability_score"]
        < real_result["stability"]["stability_score"]
    )


# This test keeps variable run timings out of DVC-tracked evaluation reports.
def test_evaluation_report_ignores_run_specific_timings() -> None:
    frame = _build_frame()
    contract = replace(
        load_contract(Path("configs/contracts/normal_operations.yaml")),
        target_delay_proportion=0.5,
    )
    sampling = {
        "model_type": "gaussian_copula",
        "source_sha256": "source-hash",
        "attempted_rows": len(frame),
        "acceptance_rate": 1.0,
    }

    first = evaluate_candidate(frame, frame, frame.copy(), contract, sampling)
    sampling["training_seconds"] = 100.0
    sampling["generation_seconds"] = 50.0
    sampling["run_id"] = "different-run-id"
    second = evaluate_candidate(frame, frame, frame.copy(), contract, sampling)

    assert first == second
    assert "training_seconds" not in first["operational"]
    assert "generation_seconds" not in first["operational"]
