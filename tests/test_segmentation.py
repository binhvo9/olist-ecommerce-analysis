"""Tests for src/segmentation.py"""

import numpy as np
import pandas as pd

from src.segmentation import build_rfm, fit_kmeans

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_df(**overrides) -> pd.DataFrame:
    """Minimal master-like DataFrame with two delivered orders for one customer."""
    base = {
        "customer_unique_id": ["cust_a", "cust_a"],
        "order_id": ["o1", "o2"],
        "order_status": ["delivered", "delivered"],
        "order_purchase_timestamp": [
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2018-06-01"),
        ],
        "total_payment": [100.0, 200.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Test 1 – build_rfm returns expected columns
# ---------------------------------------------------------------------------


def test_build_rfm_columns():
    """build_rfm must return exactly [customer_unique_id, recency, frequency, monetary]."""
    df = _make_minimal_df()
    rfm = build_rfm(df)

    expected_cols = {"customer_unique_id", "recency", "frequency", "monetary"}
    assert expected_cols == set(
        rfm.columns
    ), f"Expected columns {expected_cols}, got {set(rfm.columns)}"


# ---------------------------------------------------------------------------
# Test 2 – frequency counts unique orders per customer
# ---------------------------------------------------------------------------


def test_rfm_frequency_counts_orders():
    """Customer with 3 orders has frequency=3; customer with 1 order has frequency=1."""
    df = pd.DataFrame(
        {
            "customer_unique_id": ["cust_a", "cust_a", "cust_a", "cust_b"],
            "order_id": ["o1", "o2", "o3", "o4"],
            "order_status": ["delivered"] * 4,
            "order_purchase_timestamp": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-03-01"),
                pd.Timestamp("2018-06-01"),
                pd.Timestamp("2018-04-01"),
            ],
            "total_payment": [100.0, 150.0, 200.0, 80.0],
        }
    )

    rfm = build_rfm(df)
    rfm_indexed = rfm.set_index("customer_unique_id")

    assert (
        rfm_indexed.loc["cust_a", "frequency"] == 3
    ), f"Expected cust_a frequency=3, got {rfm_indexed.loc['cust_a', 'frequency']}"
    assert (
        rfm_indexed.loc["cust_b", "frequency"] == 1
    ), f"Expected cust_b frequency=1, got {rfm_indexed.loc['cust_b', 'frequency']}"


# ---------------------------------------------------------------------------
# Test 3 – fit_kmeans adds 'cluster' column with valid values
# ---------------------------------------------------------------------------


def test_fit_kmeans_adds_cluster():
    """fit_kmeans must add a 'cluster' column with values in range(k)."""
    rng = np.random.default_rng(0)
    n = 30
    k = 4

    # Build a synthetic rfm DataFrame with 30 customers
    rfm_raw = pd.DataFrame(
        {
            "customer_unique_id": [f"c{i}" for i in range(n)],
            "recency": rng.integers(1, 365, size=n).astype(float),
            "frequency": rng.integers(1, 10, size=n).astype(float),
            "monetary": rng.uniform(50, 2000, size=n),
        }
    )

    rfm_out, scaler, kmeans_model = fit_kmeans(rfm_raw, k=k, random_state=42)

    assert (
        "cluster" in rfm_out.columns
    ), "'cluster' column not found in output DataFrame"
    assert len(rfm_out) == n, "Row count should not change after fit_kmeans"

    unique_clusters = set(rfm_out["cluster"].unique())
    valid_clusters = set(range(k))
    assert unique_clusters.issubset(
        valid_clusters
    ), f"Cluster values {unique_clusters} are not a subset of {valid_clusters}"
    # All k clusters should be used (with 30 rows and k=4 this should always hold)
    assert (
        unique_clusters == valid_clusters
    ), f"Expected all {k} clusters to be assigned, but got {unique_clusters}"
