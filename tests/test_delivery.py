"""
Unit tests for src/delivery.py — Angle 1 delivery performance analysis.
"""

import pandas as pd
import pytest

from src.delivery import delivery_kpis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(delays, statuses=None):
    """Build a minimal DataFrame compatible with delivery_kpis."""
    n = len(delays)
    if statuses is None:
        statuses = ["delivered"] * n
    return pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(n)],
            "order_status": statuses,
            "delivery_delay": delays,
            "customer_state": ["SP"] * n,
            "review_score": [4] * n,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_delivery_kpis_keys():
    """delivery_kpis must return a dict containing all 5 required keys."""
    df = _make_df([-3, 0, 5, 10])
    result = delivery_kpis(df)

    expected_keys = {
        "on_time_rate",
        "median_delay_days",
        "mean_delay_days",
        "pct_very_late",
        "total_delivered",
    }
    assert set(result.keys()) == expected_keys


def test_on_time_calculation():
    """on_time_rate should be 0.5 when exactly half of delivered orders are on time."""
    # delays: -5 (on time), 0 (on time), 3 (late), 10 (late)  →  2/4 = 0.5
    df = _make_df([-5, 0, 3, 10])
    result = delivery_kpis(df)

    assert result["on_time_rate"] == pytest.approx(0.5)
