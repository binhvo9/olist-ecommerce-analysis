from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

TABLES = [
    "olist_orders_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_customers_dataset",
    "olist_sellers_dataset",
    "olist_products_dataset",
    "olist_geolocation_dataset",
    "product_category_name_translation",
]


def load_raw() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name in TABLES:
        path = DATA_DIR / f"{name}.csv"
        dfs[name] = pd.read_csv(path)
        print(f"{name}: {dfs[name].shape}")
    return dfs


def build_master(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = dfs["olist_orders_dataset"]
    items = dfs["olist_order_items_dataset"]
    payments = dfs["olist_order_payments_dataset"]
    reviews = dfs["olist_order_reviews_dataset"]
    customers = dfs["olist_customers_dataset"]
    sellers = dfs["olist_sellers_dataset"]
    products = dfs["olist_products_dataset"]
    translation = dfs["product_category_name_translation"]

    # aggregate payments to one row per order
    pay_agg = (
        payments.groupby("order_id")
        .agg(
            total_payment=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            primary_payment_type=("payment_type", lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    # aggregate items to one row per order
    items_agg = (
        items.groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
        .reset_index()
    )
    # keep first product_id per order for product-level features
    items_first = items[["order_id", "product_id", "seller_id"]].drop_duplicates("order_id")

    # translate product categories
    products = products.merge(translation, on="product_category_name", how="left")

    print(f"\nBuilding master dataframe:")
    df = orders.copy()
    print(f"  orders: {len(df)}")

    df = df.merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="left")
    print(f"  + customers: {len(df)}")

    df = df.merge(pay_agg, on="order_id", how="left")
    print(f"  + payments: {len(df)}")

    df = df.merge(items_agg, on="order_id", how="left")
    print(f"  + items: {len(df)}")

    df = df.merge(items_first, on="order_id", how="left")
    print(f"  + items (first product): {len(df)}")

    df = df.merge(
        products[["product_id", "product_category_name_english", "product_weight_g",
                  "product_length_cm", "product_height_cm", "product_width_cm"]],
        on="product_id", how="left"
    )
    print(f"  + products: {len(df)}")

    df = df.merge(
        sellers[["seller_id", "seller_state"]],
        on="seller_id", how="left"
    )
    print(f"  + sellers: {len(df)}")

    df = df.merge(
        reviews[["order_id", "review_score", "review_creation_date", "review_answer_timestamp"]],
        on="order_id", how="left"
    )
    print(f"  + reviews: {len(df)}")

    return df
