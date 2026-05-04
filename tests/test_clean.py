import pandas as pd
from src.clean import parse_timestamps, add_delivery_features, flag_anomalies


def _sample_orders():
    return pd.DataFrame(
        {
            "order_purchase_timestamp": ["2018-01-01 10:00:00", "2018-02-01 12:00:00"],
            "order_approved_at": ["2018-01-01 11:00:00", "2018-02-01 13:00:00"],
            "order_delivered_carrier_date": ["2018-01-05 08:00:00", "2018-02-05 08:00:00"],
            "order_delivered_customer_date": ["2018-01-10 14:00:00", "2018-02-08 10:00:00"],
            "order_estimated_delivery_date": ["2018-01-12 00:00:00", "2018-02-10 00:00:00"],
        }
    )


def test_parse_timestamps_converts_to_datetime():
    df = parse_timestamps(_sample_orders())
    assert pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(df["order_delivered_customer_date"])


def test_delivery_delay_sign():
    df = parse_timestamps(_sample_orders())
    df = add_delivery_features(df)
    # row 0: delivered Jan 10, estimated Jan 12 → early (negative delay)
    assert df.loc[0, "delivery_delay"] < 0
    # row 1: delivered Feb 3, estimated Feb 10 → early (negative delay)
    assert df.loc[1, "delivery_delay"] < 0


def test_days_to_deliver_positive():
    df = parse_timestamps(_sample_orders())
    df = add_delivery_features(df)
    assert (df["days_to_deliver"] > 0).all()


def test_flag_impossible_delivery():
    df = parse_timestamps(_sample_orders())
    df = flag_anomalies(df)
    # neither row has impossible delivery in sample
    assert df["flag_impossible_delivery"].sum() == 0
