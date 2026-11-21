"""Apply contract release gates and explain every release decision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    measured: float | str
    threshold: float | str
    reason: str
