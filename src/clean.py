import pandas as pd
import numpy as np

TIMESTAMP_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TIMESTAMP_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def add_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["days_to_deliver"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["delivery_delay"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days  # positive = late, negative = early
    df["days_to_approve"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600  # hours
    return df


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )
    df["freight_ratio"] = df["total_freight"] / (df["total_price"] + df["total_freight"] + 1e-9)
    return df


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # delivered before shipped
    df["flag_impossible_delivery"] = (
        df["order_delivered_customer_date"] < df["order_delivered_carrier_date"]
    )
    # review created before delivery
    if "review_creation_date" in df.columns:
        df["review_creation_date"] = pd.to_datetime(df["review_creation_date"])
        df["flag_early_review"] = (
            df["review_creation_date"] < df["order_delivered_customer_date"]
        )
    return df


def filter_delivered(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only delivered orders for delivery/review analyses."""
    delivered = df[df["order_status"] == "delivered"].copy()
    print(f"Delivered orders: {len(delivered)} / {len(df)} ({len(delivered)/len(df):.1%})")
    return delivered


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_timestamps(df)
    df = add_delivery_features(df)
    df = add_product_features(df)
    df = flag_anomalies(df)
    return df
