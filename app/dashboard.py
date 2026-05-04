"""Streamlit dashboard — Olist Brazilian E-Commerce Analysis."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Olist E-Commerce Analysis",
    page_icon="🛒",
    layout="wide",
)

REPORTS = Path(__file__).parent.parent / "reports"
DATA = Path(__file__).parent.parent / "data"


@st.cache_data
def load_master():
    path = REPORTS / "master.parquet"
    if not path.exists():
        st.error("Run the analysis notebook first to generate reports/master.parquet")
        st.stop()
    return pd.read_parquet(path)


def kpi_row(df):
    delivered = df[df["order_status"] == "delivered"]
    on_time = (delivered["delivery_delay"] <= 0).mean()
    avg_score = delivered["review_score"].mean()
    gmv = delivered["total_payment"].sum()
    n_customers = delivered["customer_unique_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total GMV", f"R${gmv/1e6:.1f}M")
    c2.metric("On-Time Delivery", f"{on_time:.1%}")
    c3.metric("Avg Review Score", f"{avg_score:.2f} / 5")
    c4.metric("Unique Customers", f"{n_customers:,}")


def tab_overview(df):
    st.subheader("Monthly Order Volume")
    monthly = (
        df.set_index("order_purchase_timestamp")
        .resample("ME")["order_id"]
        .count()
        .reset_index()
    )
    monthly.columns = ["month", "orders"]
    fig = px.line(monthly, x="month", y="orders", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Categories by Revenue")
    cat = (
        df.groupby("product_category_name_english")["total_payment"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    cat.columns = ["category", "revenue"]
    fig2 = px.bar(cat, x="revenue", y="category", orientation="h", color="revenue",
                  color_continuous_scale="Blues")
    st.plotly_chart(fig2, use_container_width=True)


def tab_delivery(df):
    delivered = df[df["order_status"] == "delivered"].copy()

    st.subheader("Delivery Delay Distribution")
    fig = px.histogram(
        delivered.assign(delay=delivered["delivery_delay"].clip(-20, 60)),
        x="delay", nbins=60, color_discrete_sequence=["steelblue"],
        labels={"delay": "Days (negative = early)"},
    )
    fig.add_vline(x=0, line_color="red", line_dash="dash", annotation_text="On time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean Delivery Delay by State")
    state_delay = (
        delivered.groupby("customer_state")["delivery_delay"]
        .mean()
        .reset_index()
    )
    state_delay.columns = ["state", "mean_delay"]
    fig2 = px.bar(
        state_delay.sort_values("mean_delay", ascending=False),
        x="state", y="mean_delay",
        color="mean_delay", color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Delay vs Review Score")
    score_delay = delivered.groupby("review_score")["delivery_delay"].mean().reset_index()
    fig3 = px.bar(score_delay, x="review_score", y="delivery_delay",
                  color="delivery_delay", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig3, use_container_width=True)


def tab_segments(df):
    seg_path = REPORTS / "rfm_segments.parquet"
    if not seg_path.exists():
        st.info("RFM segments not yet generated. Run the analysis notebook.")
        return

    rfm = pd.read_parquet(seg_path)

    st.subheader("Customer Segment Distribution")
    counts = rfm["segment"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    fig = px.pie(counts, names="segment", values="count", hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RFM Profile by Segment")
    profile = rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1)
    st.dataframe(profile.style.background_gradient(cmap="YlOrRd"), use_container_width=True)


def tab_predictor(df):
    st.subheader("Review Risk Predictor")
    st.info("Train the model first (Angle 3 notebook section) to enable live predictions.")

    model_path = REPORTS / "review_model.joblib"
    if not model_path.exists():
        st.warning("Model artifact not found at reports/review_model.joblib")
        return

    import joblib
    model = joblib.load(model_path)

    col1, col2 = st.columns(2)
    with col1:
        delay = st.slider("Delivery delay (days)", -10, 30, 0)
        price = st.number_input("Order price (R$)", 10.0, 5000.0, 150.0)
        freight = st.number_input("Freight value (R$)", 0.0, 500.0, 20.0)
        installments = st.slider("Payment installments", 1, 12, 1)

    with col2:
        category = st.selectbox("Product category", ["bed_bath_table", "health_beauty",
                                                      "computers_accessories", "furniture_decor",
                                                      "sports_leisure", "other"])
        customer_state = st.selectbox("Customer state", ["SP", "RJ", "MG", "RS", "PR", "other"])

    if st.button("Predict Risk"):
        import numpy as np
        row = pd.DataFrame([{
            "delivery_delay": delay, "days_to_deliver": max(delay + 7, 1),
            "days_to_approve": 1.0, "total_price": price,
            "total_freight": freight, "freight_ratio": freight / (price + freight + 1e-9),
            "payment_installments": installments, "n_items": 1,
            "product_weight_g": 500, "product_volume_cm3": 1000,
            "customer_state": customer_state, "seller_state": "SP",
            "primary_payment_type": "credit_card",
            "product_category_name_english": category,
        }])
        prob = model.predict_proba(row)[0][1]
        color = "red" if prob > 0.5 else "orange" if prob > 0.25 else "green"
        band = "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.25 else "LOW"
        st.markdown(
            f"<h2 style='color:{color}'>Risk: {band} ({prob:.1%})</h2>",
            unsafe_allow_html=True,
        )


def main():
    st.title("🛒 Olist Brazilian E-Commerce Analysis")
    st.caption("100k orders · 2016–2018 · 8 relational tables")

    df = load_master()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    kpi_row(df)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", "🚚 Delivery Performance", "👥 Customer Segments", "🔮 Review Predictor"
    ])

    with tab1:
        tab_overview(df)
    with tab2:
        tab_delivery(df)
    with tab3:
        tab_segments(df)
    with tab4:
        tab_predictor(df)


if __name__ == "__main__":
    main()
