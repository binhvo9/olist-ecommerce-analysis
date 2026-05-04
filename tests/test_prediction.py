"""Tests for src/prediction.py"""

import numpy as np
import pandas as pd

from src.prediction import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_df,
    build_preprocessor,
    train,
)


def _make_master_df(n=200, random_state=42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    return pd.DataFrame(
        {
            "order_status": ["delivered"] * n,
            "review_score": rng.integers(1, 6, size=n),
            "delivery_delay": rng.uniform(-5, 20, size=n),
            "days_to_deliver": rng.integers(3, 30, size=n).astype(float),
            "days_to_approve": rng.uniform(0.5, 24, size=n),
            "total_price": rng.uniform(20, 500, size=n),
            "total_freight": rng.uniform(5, 80, size=n),
            "freight_ratio": rng.uniform(0.05, 0.4, size=n),
            "payment_installments": rng.integers(1, 12, size=n).astype(float),
            "n_items": rng.integers(1, 5, size=n).astype(float),
            "product_weight_g": rng.uniform(100, 5000, size=n),
            "product_volume_cm3": rng.uniform(500, 20000, size=n),
            "customer_state": rng.choice(["SP", "RJ", "MG", "RS"], size=n),
            "seller_state": rng.choice(["SP", "MG"], size=n),
            "primary_payment_type": rng.choice(
                ["credit_card", "boleto", "voucher"], size=n
            ),
            "product_category_name_english": rng.choice(
                ["bed_bath_table", "electronics", "fashion"], size=n
            ),
        }
    )


def test_build_model_df_drops_neutral():
    df = _make_master_df(300)
    model_df = build_model_df(df)
    assert 3 not in model_df["review_score"].values


def test_build_model_df_binary_target():
    df = _make_master_df(300)
    model_df = build_model_df(df)
    assert set(model_df["bad_review"].unique()).issubset({0, 1})
    assert (model_df["bad_review"] == 1).equals(model_df["review_score"] <= 2)


def test_preprocessor_output_shape():
    df = _make_master_df(100)
    model_df = build_model_df(df)
    X = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > len(NUMERIC_FEATURES)


def test_train_returns_expected_keys():
    df = _make_master_df(300)
    results = train(df, test_size=0.2, random_state=42)
    for key in ("rf_pipeline", "lr_pipeline", "rf_auc", "lr_auc", "X_test", "y_test"):
        assert key in results, f"Missing key: {key}"


def test_train_auc_above_chance():
    df = _make_master_df(500)
    results = train(df, test_size=0.2, random_state=42)
    assert results["rf_auc"] > 0.5
    assert results["lr_auc"] > 0.5
