"""Load scenario contracts and evaluate their mandatory business rules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import yaml

from synthgate.data import SAFE_COLUMNS

RULE_NAMES: Final = {
    "no_nulls",
    "valid_delay_status",
    "valid_shipping_days",
    "valid_discount_rate",
    "positive_quantity",
    "positive_values",
    "valid_order_shipping_interval",
}


@dataclass(frozen=True)
class ScenarioContract:
    name: str
    target_delay_proportion: float
    target_tolerance: float
    requested_rows: int
    mutable_columns: tuple[str, ...]
    stable_columns: tuple[str, ...]
    stable_relationships: tuple[tuple[str, str], ...]
    mandatory_rules: tuple[str, ...]
    max_scenario_error: float
    minimum_stability_score: float
    maximum_exact_copy_rate: float
    max_attempts: int
    batch_size: int
    sha256: str


# This function validates contract configuration before any model uses it.
def load_contract(path: Path) -> ScenarioContract:
    if not path.is_file():
        raise FileNotFoundError(f"Scenario contract not found: {path}")
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("Scenario contract must be a YAML mapping")

    required = {
        "name",
        "target_delay_proportion",
        "target_tolerance",
        "requested_rows",
        "mutable_columns",
        "stable_columns",
        "stable_relationships",
        "mandatory_rules",
        "quality_thresholds",
        "sampling",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Scenario contract is missing fields: {', '.join(missing)}")

    quality = raw["quality_thresholds"]
    sampling = raw["sampling"]
    if not isinstance(quality, dict) or not isinstance(sampling, dict):
        raise ValueError("quality_thresholds and sampling must be YAML mappings")

    target = float(raw["target_delay_proportion"])
    tolerance = float(raw["target_tolerance"])
    requested_rows = int(raw["requested_rows"])
    mutable = tuple(raw["mutable_columns"])
    stable = tuple(raw["stable_columns"])
    rules = tuple(raw["mandatory_rules"])
    relationships = tuple(tuple(pair) for pair in raw["stable_relationships"])

    if not 0 <= target <= 1 or not 0 <= tolerance <= 1:
        raise ValueError("Scenario delay target and tolerance must be between 0 and 1")
    if requested_rows < 1:
        raise ValueError("requested_rows must be greater than zero")
    if set(mutable).intersection(stable):
        raise ValueError("A column cannot be both mutable and stable")
    if set(mutable).union(stable) != set(SAFE_COLUMNS):
        raise ValueError("Contract must classify every safe feature as mutable or stable")
    if set(rules).difference(RULE_NAMES):
        raise ValueError(f"Contract contains unknown mandatory rules: {set(rules) - RULE_NAMES}")
    if not set(mutable).issubset(SAFE_COLUMNS):
        raise ValueError("Contract contains unknown mutable columns")
    if not set(stable).issubset(SAFE_COLUMNS):
        raise ValueError("Contract contains unknown stable columns")
    if any(len(pair) != 2 or not set(pair).issubset(stable) for pair in relationships):
        raise ValueError("Stable relationships must reference two stable columns")

    max_scenario_error = float(quality["max_scenario_error"])
    minimum_stability_score = float(quality["minimum_stability_score"])
    maximum_exact_copy_rate = float(quality["maximum_exact_copy_rate"])
    if any(
        not 0 <= value <= 1
        for value in (max_scenario_error, minimum_stability_score, maximum_exact_copy_rate)
    ):
        raise ValueError("Quality thresholds must be between 0 and 1")

    max_attempts = int(sampling["max_attempts"])
    batch_size = int(sampling["batch_size"])
    if max_attempts < 1 or batch_size < 1:
        raise ValueError("Sampling attempts and batch size must be greater than zero")

    return ScenarioContract(
        name=str(raw["name"]),
        target_delay_proportion=target,
        target_tolerance=tolerance,
        requested_rows=requested_rows,
        mutable_columns=mutable,
        stable_columns=stable,
        stable_relationships=relationships,
        mandatory_rules=rules,
        max_scenario_error=max_scenario_error,
        minimum_stability_score=minimum_stability_score,
        maximum_exact_copy_rate=maximum_exact_copy_rate,
        max_attempts=max_attempts,
        batch_size=batch_size,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _rule_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "no_nulls": frame.notna().all(axis=1),
        "valid_delay_status": frame["late_delivery_risk"].isin([0, 1])
        & ((frame["late_delivery_risk"] == 1) == (frame["delivery_status"] == "Late delivery")),
        "valid_shipping_days": frame["days_for_shipping"].between(0, 10)
        & frame["days_for_shipment_scheduled"].between(0, 10),
        "valid_discount_rate": frame["discount_rate"].between(0, 1),
        "positive_quantity": frame["quantity"] > 0,
        "positive_values": (frame["product_price"] > 0)
        & (frame["sales"] > 0)
        & (frame["order_item_total"] > 0),
        "valid_order_shipping_interval": frame["order_to_ship_days"] > 0,
    }


def _check_features(frame: pd.DataFrame) -> None:
    missing = sorted(set(SAFE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required features: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")


# This function filters invalid synthetic rows before they can be released.
def valid_rows_mask(frame: pd.DataFrame, contract: ScenarioContract) -> pd.Series:
    _check_features(frame)
    rules = _rule_masks(frame)
    valid = pd.Series(True, index=frame.index)
    for rule in contract.mandatory_rules:
        valid &= rules[rule].fillna(False)
    return valid


# This function reports row-level violation counts and rates for the contract rules.
def evaluate_rules(frame: pd.DataFrame, contract: ScenarioContract) -> dict[str, dict[str, float]]:
    _check_features(frame)
    rules = _rule_masks(frame)
    results: dict[str, dict[str, float]] = {}
    row_count = len(frame)
    for rule in contract.mandatory_rules:
        passed = rules[rule].fillna(False)
        violations = int((~passed).sum())
        results[rule] = {
            "violations": violations,
            "violation_rate": violations / row_count,
        }
    return results
