from synthgate.registry import ComparisonResult, promote_candidate, rollback_model
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


class FakeClient:
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[f"{name}:{alias}"] = version


def test_promotion_updates_approved_alias_after_passed_comparison() -> None:
    client = FakeClient()
    comparison = ComparisonResult(True, "Candidate meets thresholds.", 0.70, 0.70)

    result = promote_candidate(client, "SynthGate-CTGAN", "2", comparison)

    assert result["decision"] == "PROMOTED"
    assert client.aliases["SynthGate-CTGAN:approved"] == "2"


def test_promotion_blocks_failed_comparison() -> None:
    client = FakeClient()
    comparison = ComparisonResult(False, "Candidate stability regressed.", 0.60, 0.70)

    try:
        promote_candidate(client, "SynthGate-CTGAN", "2", comparison)
        raise AssertionError("Failed comparison should block promotion")
    except RuntimeError as error:
        assert "stability" in str(error)


def test_rollback_repoints_approved_alias() -> None:
    client = FakeClient()

    result = rollback_model(client, "SynthGate-CTGAN", "1")

    assert result["decision"] == "ROLLED_BACK"
    assert client.aliases["SynthGate-CTGAN:approved"] == "1"
