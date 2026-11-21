from synthgate.registry import compare_candidate


def _evaluation(stability: float, scenario_error: float, copy_rate: float) -> dict[str, object]:
    return {
        "stability": {"stability_score": stability},
        "scenario_target_error": scenario_error,
        "similarity_risk_indicators": {"training_exact_copy_rate": copy_rate},
    }


def test_candidate_is_approved_when_metrics_do_not_regress() -> None:
    approved = _evaluation(0.70, 0.01, 0.01)
    candidate = _evaluation(0.69, 0.01, 0.01)

    result = compare_candidate(candidate, approved)

    assert result.passed
    assert "meets" in result.reason


def test_candidate_is_blocked_when_stability_regresses() -> None:
    approved = _evaluation(0.70, 0.01, 0.01)
    candidate = _evaluation(0.60, 0.01, 0.01)

    result = compare_candidate(candidate, approved)

    assert not result.passed
    assert "stability" in result.reason


def test_candidate_is_blocked_when_scenario_error_increases() -> None:
    approved = _evaluation(0.70, 0.01, 0.01)
    candidate = _evaluation(0.70, 0.05, 0.01)

    result = compare_candidate(candidate, approved)

    assert not result.passed
    assert "scenario" in result.reason


def test_candidate_is_blocked_when_exact_copy_risk_increases() -> None:
    approved = _evaluation(0.70, 0.01, 0.01)
    candidate = _evaluation(0.70, 0.01, 0.05)

    result = compare_candidate(candidate, approved)

    assert not result.passed
    assert "exact-copy" in result.reason
