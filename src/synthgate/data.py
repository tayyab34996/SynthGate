"""Ingest, de-identify, validate, and split the shipment dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Final

import pandas as pd
import pandera.pandas as pa
import yaml
from sklearn.model_selection import train_test_split

SOURCE_COLUMNS: Final = {
    "Type": "payment_type",
    "Days for shipping (real)": "days_for_shipping",
    "Days for shipment (scheduled)": "days_for_shipment_scheduled",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Category Name": "category",
    "Customer Segment": "customer_segment",
    "Department Name": "department",
    "Market": "market",
    "Order Region": "order_region",
    "Order Status": "order_status",
    "Shipping Mode": "shipping_mode",
    "Order Item Discount Rate": "discount_rate",
    "Order Item Quantity": "quantity",
    "Order Item Product Price": "product_price",
    "Sales": "sales",
    "Order Item Total": "order_item_total",
    "Benefit per order": "benefit_per_order",
    "Order Item Profit Ratio": "profit_ratio",
}

NUMERIC_COLUMNS: Final = (
    "days_for_shipping",
    "days_for_shipment_scheduled",
    "late_delivery_risk",
    "discount_rate",
    "quantity",
    "product_price",
    "sales",
    "order_item_total",
    "benefit_per_order",
    "profit_ratio",
    "order_to_ship_days",
)

CATEGORICAL_COLUMNS: Final = (
    "payment_type",
    "delivery_status",
    "category",
    "customer_segment",
    "department",
    "market",
    "order_region",
    "order_status",
    "shipping_mode",
)

SAFE_COLUMNS: Final = (*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS)

SHIPMENT_SCHEMA: Final = pa.DataFrameSchema(
    {
        **{
            name: pa.Column(str, nullable=False, checks=pa.Check.str_length(min_value=1))
            for name in CATEGORICAL_COLUMNS
        },
        "days_for_shipping": pa.Column(int, nullable=False, checks=pa.Check.in_range(0, 10)),
        "days_for_shipment_scheduled": pa.Column(
            int, nullable=False, checks=pa.Check.in_range(0, 10)
        ),
        "late_delivery_risk": pa.Column(int, nullable=False, checks=pa.Check.isin([0, 1])),
        "discount_rate": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 1)),
        "quantity": pa.Column(int, nullable=False, checks=pa.Check.greater_than(0)),
        "product_price": pa.Column(float, nullable=False, checks=pa.Check.greater_than(0)),
        "sales": pa.Column(float, nullable=False, checks=pa.Check.greater_than(0)),
        "order_item_total": pa.Column(float, nullable=False, checks=pa.Check.greater_than(0)),
        "benefit_per_order": pa.Column(float, nullable=False),
        "profit_ratio": pa.Column(float, nullable=False),
        "order_to_ship_days": pa.Column(float, nullable=False, checks=pa.Check.greater_than(0)),
    },
    strict=True,
    coerce=True,
    checks=pa.Check(
        lambda frame: (
            (frame["late_delivery_risk"] == 1) == (frame["delivery_status"] == "Late delivery")
        ),
        error="late_delivery_risk must match the delivery status",
    ),
)


# This function keeps only modeling features and derives elapsed time, excluding direct identifiers.
def deidentify_source(raw: pd.DataFrame) -> pd.DataFrame:
    required = {*SOURCE_COLUMNS, "order date (DateOrders)", "shipping date (DateOrders)"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Source dataset is missing required columns: {', '.join(missing)}")

    frame = raw.loc[:, list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).copy()
    order_time = pd.to_datetime(raw["order date (DateOrders)"], errors="coerce")
    shipping_time = pd.to_datetime(raw["shipping date (DateOrders)"], errors="coerce")
    if order_time.isna().any() or shipping_time.isna().any():
        raise ValueError("Order and shipping dates must be valid timestamps")

    elapsed = (shipping_time - order_time).dt.total_seconds() / 86400
    if elapsed.lt(0).any():
        raise ValueError("Shipping timestamps cannot precede order timestamps")
    frame["order_to_ship_days"] = elapsed

    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("string")
    return frame.loc[:, SAFE_COLUMNS]


# This function validates the retained columns before they enter the training pipeline.
def validate_shipments(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Shipment dataset must contain at least one row")
    try:
        return SHIPMENT_SCHEMA.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as error:
        failures = error.failure_cases[["column", "check", "failure_case"]]
        details = failures.head(10).to_dict(orient="records")
        raise ValueError(f"Shipment validation failed: {details}") from error


# This function creates repeatable train and holdout files with class balance preserved.
def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    encoding: str = "latin1",
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if not input_path.is_file():
        raise FileNotFoundError(f"Source dataset not found: {input_path}")

    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    raw = pd.read_csv(input_path, encoding=encoding)
    clean = validate_shipments(deidentify_source(raw))
    if clean["late_delivery_risk"].nunique() < 2:
        raise ValueError("Source dataset must contain both delivery-risk classes")
    class_counts = clean["late_delivery_risk"].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each delivery-risk class must contain at least two rows")
    holdout_rows = math.ceil(len(clean) * holdout_fraction)
    class_count = len(class_counts)
    if holdout_rows < class_count or len(clean) - holdout_rows < class_count:
        raise ValueError("Train and holdout splits must each contain every delivery-risk class")

    train, holdout = train_test_split(
        clean,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=clean["late_delivery_risk"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_dir / "shipments.csv", index=False)
    train.to_csv(output_dir / "train.csv", index=False)
    holdout.to_csv(output_dir / "holdout.csv", index=False)

    profile: dict[str, object] = {
        "source_sha256": source_hash,
        "rows": int(len(clean)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "columns": list(clean.columns),
        "nulls": {name: int(count) for name, count in clean.isna().sum().items()},
        "late_delivery_rate": float(clean["late_delivery_risk"].mean()),
        "seed": seed,
        "holdout_fraction": holdout_fraction,
    }
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


# This command-line entry point runs the deterministic data preparation stage.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = config["source"]
    split = config["split"]
    profile = prepare_dataset(
        input_path=Path(source["input"]),
        output_dir=Path("data/processed"),
        encoding=source["encoding"],
        holdout_fraction=split["holdout_fraction"],
        seed=split["seed"],
    )
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
