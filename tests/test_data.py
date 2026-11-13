import pandas as pd
import pytest

from synthgate.data import SAFE_COLUMNS, deidentify_source, prepare_dataset, validate_shipments


@pytest.fixture
def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["DEBIT", "CASH", "DEBIT", "PAYMENT", "CASH"],
            "Days for shipping (real)": [2, 1, 3, 2, 1],
            "Days for shipment (scheduled)": [2, 2, 3, 2, 1],
            "Delivery Status": [
                "Late delivery",
                "Shipping on time",
                "Advance shipping",
                "Shipping canceled",
                "Late delivery",
            ],
            "Late_delivery_risk": [1, 0, 0, 0, 1],
            "Category Name": ["A", "B", "A", "C", "B"],
            "Customer Segment": ["Consumer"] * 5,
            "Department Name": ["D"] * 5,
            "Market": ["M"] * 5,
            "Order Region": ["R"] * 5,
            "Order Status": ["COMPLETE"] * 5,
            "Shipping Mode": ["Standard Class"] * 5,
            "Order Item Discount Rate": [0.1] * 5,
            "Order Item Quantity": [1, 2, 3, 1, 2],
            "Order Item Product Price": [10.0] * 5,
            "Sales": [10.0] * 5,
            "Order Item Total": [9.0] * 5,
            "Benefit per order": [1.0] * 5,
            "Order Item Profit Ratio": [0.1] * 5,
            "order date (DateOrders)": ["2017-01-01"] * 5,
            "shipping date (DateOrders)": [
                "2017-01-03",
                "2017-01-02",
                "2017-01-04",
                "2017-01-03",
                "2017-01-02",
            ],
            "Customer Email": ["person@example.com"] * 5,
            "Customer Password": ["secret"] * 5,
        }
    )


# This test ensures direct identifiers never enter the retained training features.
def test_deidentify_source_keeps_only_safe_features(source_frame: pd.DataFrame) -> None:
    clean = deidentify_source(source_frame)

    assert tuple(clean.columns) == SAFE_COLUMNS
    assert "Customer Email" not in clean
    assert "Customer Password" not in clean
    assert clean["order_to_ship_days"].tolist() == [2.0, 1.0, 3.0, 2.0, 1.0]


# This test checks the delivery flag and timestamp rules used before training.
def test_validate_shipments_rejects_inconsistent_delivery_flag(
    source_frame: pd.DataFrame,
) -> None:
    clean = deidentify_source(source_frame)
    clean.loc[0, "late_delivery_risk"] = 0

    with pytest.raises(ValueError, match="Shipment validation failed"):
        validate_shipments(clean)


# This test verifies deterministic stratified output and the expected split sizes.
def test_prepare_dataset_is_reproducible(
    source_frame: pd.DataFrame,
    tmp_path,
) -> None:
    input_path = tmp_path / "source.csv"
    source_frame.to_csv(input_path, index=False)

    first = prepare_dataset(input_path, tmp_path / "first", holdout_fraction=0.4, seed=7)
    second = prepare_dataset(input_path, tmp_path / "second", holdout_fraction=0.4, seed=7)

    assert first == second
    assert first["rows"] == 5
    assert first["train_rows"] == 3
    assert first["holdout_rows"] == 2
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "first" / "train.csv"),
        pd.read_csv(tmp_path / "second" / "train.csv"),
    )
