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


def promote_candidate(
    client: object,
    model_name: str,
    candidate_version: str,
    comparison: ComparisonResult,
) -> dict[str, object]:
    """This function promotes only candidates that pass release comparison."""
    if not comparison.passed:
        raise RuntimeError(comparison.reason)
    client.set_registered_model_alias(model_name, "approved", str(candidate_version))
    return {
        "model_name": model_name,
        "approved_version": str(candidate_version),
        "alias": "approved",
        "decision": "PROMOTED",
        "reason": comparison.reason,
    }


def rollback_model(
    client: object,
    model_name: str,
    previous_version: str,
) -> dict[str, object]:
    """This function repoints the approved alias to a previous version."""
    client.set_registered_model_alias(model_name, "approved", str(previous_version))
    return {
        "model_name": model_name,
        "approved_version": str(previous_version),
        "alias": "approved",
        "decision": "ROLLED_BACK",
        "reason": "Approved alias restored to the previous model version.",
    }
