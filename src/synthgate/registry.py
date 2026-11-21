"""Compare, promote, and roll back registered generator models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    reason: str
    candidate_metric: float
    approved_metric: float
