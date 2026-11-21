"""Apply contract release gates and explain every release decision."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from synthgate.data import SAFE_COLUMNS


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    measured: float | str
    threshold: float | str
    reason: str


def evaluate_release_gates(
    candidate: pd.DataFrame,
    evaluation: dict[str, object],
) -> list[GateResult]:
    """This function evaluates every release gate and keeps failures explicit."""
    contract_name = str(evaluation["contract"])
    stability = evaluation["stability"]
    similarity = evaluation["similarity_risk_indicators"]
    rules = evaluation["rule_results"]

    missing = sorted(set(SAFE_COLUMNS).difference(candidate.columns))
    extra = sorted(set(candidate.columns).difference(SAFE_COLUMNS))
    schema_passed = candidate.notna().all().all() and not missing and not extra
    schema_result = "valid" if schema_passed else f"missing={missing}, extra={extra}"
    schema = GateResult(
        "G1_SCHEMA",
        schema_passed,
        schema_result,
        "No missing or extra columns and no nulls",
        "Schema is valid." if schema_passed else "Schema validation failed.",
    )

    violations = sum(int(rule["violations"]) for rule in rules.values())
    rules_passed = violations == 0
    rules_gate = GateResult(
        "G2_RULES",
        rules_passed,
        violations,
        0,
        "All mandatory business rules passed."
        if rules_passed
        else f"{violations} mandatory rule violations were found.",
    )

    scenario_error = float(evaluation["scenario_target_error"])
    maximum_error = float(evaluation.get("maximum_scenario_error", 0.03))
    scenario_passed = scenario_error <= maximum_error
    scenario = GateResult(
        "G3_SCENARIO",
        scenario_passed,
        scenario_error,
        maximum_error,
        f"{contract_name} target is within tolerance."
        if scenario_passed
        else f"{contract_name} target error exceeds tolerance.",
    )

    stability_score = float(stability["stability_score"])
    minimum_stability = float(evaluation.get("minimum_stability_score", 0.6))
    stability_passed = stability_score >= minimum_stability
    stability_gate = GateResult(
        "G4_STABILITY",
        stability_passed,
        stability_score,
        minimum_stability,
        "Stable properties meet the minimum score."
        if stability_passed
        else "Stable properties are below the minimum score.",
    )

    exact_copy_rate = float(similarity["training_exact_copy_rate"])
    maximum_copy_rate = float(evaluation.get("maximum_exact_copy_rate", 0.05))
    similarity_passed = exact_copy_rate <= maximum_copy_rate
    similarity_gate = GateResult(
        "G5_SIMILARITY",
        similarity_passed,
        exact_copy_rate,
        maximum_copy_rate,
        "Training exact-copy risk is below the limit."
        if similarity_passed
        else "Training exact-copy risk exceeds the limit.",
    )

    sampling_status = str(evaluation.get("sampling_status", "success"))
    operational_passed = sampling_status == "success"
    operational = GateResult(
        "G6_OPERATIONAL",
        operational_passed,
        sampling_status,
        "success",
        "Sampling completed successfully."
        if operational_passed
        else "Sampling was unsuccessful and cannot be released.",
    )
    return [schema, rules_gate, scenario, stability_gate, similarity_gate, operational]
