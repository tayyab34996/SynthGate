"""Compare, promote, and roll back registered generator models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    reason: str
    candidate_metric: float
    approved_metric: float


def compare_candidate(
    candidate_evaluation: dict[str, object],
    approved_evaluation: dict[str, object],
    stability_margin: float = 0.02,
    scenario_margin: float = 0.01,
    copy_rate_margin: float = 0.01,
) -> ComparisonResult:
    """This function blocks a candidate that regresses on key evaluation metrics."""
    candidate_stability = float(candidate_evaluation["stability"]["stability_score"])
    approved_stability = float(approved_evaluation["stability"]["stability_score"])
    candidate_error = float(candidate_evaluation["scenario_target_error"])
    approved_error = float(approved_evaluation["scenario_target_error"])
    candidate_copy = float(
        candidate_evaluation["similarity_risk_indicators"]["training_exact_copy_rate"]
    )
    approved_copy = float(
        approved_evaluation["similarity_risk_indicators"]["training_exact_copy_rate"]
    )

    if candidate_stability < approved_stability - stability_margin:
        return ComparisonResult(
            False,
            "Candidate stability regressed beyond the allowed margin.",
            candidate_stability,
            approved_stability,
        )
    if candidate_error > approved_error + scenario_margin:
        return ComparisonResult(
            False,
            "Candidate scenario target error increased beyond the allowed margin.",
            candidate_error,
            approved_error,
        )
    if candidate_copy > approved_copy + copy_rate_margin:
        return ComparisonResult(
            False,
            "Candidate exact-copy risk increased beyond the allowed margin.",
            candidate_copy,
            approved_copy,
        )
    return ComparisonResult(
        True,
        "Candidate meets the comparison thresholds.",
        candidate_stability,
        approved_stability,
    )
