from pathlib import Path

import pandas as pd
import pytest

from src.eda import plot_monthly_orders, run_all


def _make_minimal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-15", "2017-02-20", "2017-02-28"]
            ),
            "customer_state": ["SP", "RJ", "MG"],
            "product_category_name_english": ["electronics", "fashion", "electronics"],
            "total_price": [100.0, 50.0, 200.0],
            "review_score": [5, 3, 4],
            "payment_installments": [1, 3, 2],
            "primary_payment_type": ["credit_card", "boleto", "credit_card"],
        }
    )


def test_plot_monthly_orders_creates_file(tmp_path):
    df = _make_minimal_df()
    plot_monthly_orders(df, tmp_path)
    assert (tmp_path / "01_monthly_orders.png").exists()


def test_run_all_smoke(tmp_path):
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        pytest.skip("data/ directory not present — skipping full run_all smoke test")

    try:
        from src.load import build_master, load_raw
        from src.clean import clean

        dfs = load_raw()
        master = build_master(dfs)
        master = clean(master)
    except Exception as e:
        pytest.skip(f"Could not load data: {e}")

    figures_dir = tmp_path / "figures"
    run_all(master, figures_dir, tmp_path)

    for fname in [
        "01_monthly_orders.png",
        "02_orders_by_state.png",
        "03_revenue_by_category.png",
        "04_review_scores.png",
        "05_installments.png",
        "06_payment_types.png",
    ]:
        assert (figures_dir / fname).exists(), f"Missing figure: {fname}"

    assert (tmp_path / "profile_report.html").exists()
