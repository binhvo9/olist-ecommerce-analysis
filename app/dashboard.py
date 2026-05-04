"""Streamlit dashboard — Olist Brazilian E-Commerce Analysis."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Olist E-Commerce Analysis",
    page_icon="🛒",
    layout="wide",
)

REPORTS = Path(__file__).parent.parent / "reports"


@st.cache_data
def load_master() -> pd.DataFrame:
    path = REPORTS / "master.parquet"
    if not path.exists():
        st.error(
            "Run the analysis notebook first to generate **reports/master.parquet**. "
            "See README for instructions."
        )
        st.stop()
    df = pd.read_parquet(path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


@st.cache_data
def load_rfm() -> pd.DataFrame | None:
    path = REPORTS / "rfm_segments.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_resource
def load_model():
    import joblib

    path = REPORTS / "review_model.joblib"
    return joblib.load(path) if path.exists() else None


def kpi_row(df: pd.DataFrame) -> None:
    delivered = df[df["order_status"] == "delivered"]
    on_time = (delivered["delivery_delay"] <= 0).mean()
    avg_score = delivered["review_score"].mean()
    gmv = delivered["total_payment"].sum()
    n_customers = delivered["customer_unique_id"].nunique()
    pct_late = (delivered["delivery_delay"] > 7).mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total GMV", f"R${gmv / 1e6:.1f}M")
    c2.metric("On-Time Delivery", f"{on_time:.1%}")
    c3.metric("Avg Review Score", f"{avg_score:.2f} / 5")
    c4.metric("Unique Customers", f"{n_customers:,}")
    c5.metric("Very Late (>7d)", f"{pct_late:.1%}", delta_color="inverse")


def tab_overview(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Order Volume")
        monthly = (
            df.set_index("order_purchase_timestamp")
            .resample("ME")["order_id"]
            .count()
            .reset_index()
        )
        monthly.columns = ["month", "orders"]
        fig = px.line(
            monthly,
            x="month",
            y="orders",
            markers=True,
            color_discrete_sequence=["#2563EB"],
        )
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Payment Type Breakdown")
        pay = df["primary_payment_type"].value_counts().reset_index()
        pay.columns = ["type", "count"]
        fig2 = px.pie(
            pay,
            names="type",
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 15 Categories by Revenue")
    cat = (
        df.groupby("product_category_name_english")["total_payment"]
        .sum()
        .sort_values(ascending=True)
        .tail(15)
        .reset_index()
    )
    cat.columns = ["category", "revenue"]
    fig3 = px.bar(
        cat,
        x="revenue",
        y="category",
        orientation="h",
        color="revenue",
        color_continuous_scale="Blues",
        labels={"revenue": "Revenue (R$)"},
    )
    fig3.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)


def tab_delivery(df: pd.DataFrame) -> None:
    delivered = df[df["order_status"] == "delivered"].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Delay Distribution")
        clipped = delivered["delivery_delay"].clip(-20, 60)
        fig = px.histogram(
            clipped,
            nbins=60,
            color_discrete_sequence=["#2563EB"],
            labels={"value": "Days (negative = early)"},
        )
        fig.add_vline(
            x=0,
            line_color="red",
            line_dash="dash",
            annotation_text="On time",
            annotation_position="top right",
        )
        fig.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Review Score vs Delay")
        score_delay = (
            delivered.groupby("review_score")["delivery_delay"].mean().reset_index()
        )
        score_delay.columns = ["score", "mean_delay"]
        fig2 = px.bar(
            score_delay,
            x="score",
            y="mean_delay",
            color="mean_delay",
            color_continuous_scale="RdYlGn_r",
            labels={"mean_delay": "Mean delay (days)", "score": "Review score"},
        )
        fig2.add_hline(y=0, line_color="black", line_dash="dot")
        fig2.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Mean Delivery Delay by State")
    state_delay = (
        delivered.groupby("customer_state")["delivery_delay"]
        .mean()
        .reset_index()
        .sort_values("delivery_delay", ascending=False)
    )
    state_delay.columns = ["state", "mean_delay"]
    state_delay["color"] = state_delay["mean_delay"].apply(
        lambda x: "Late" if x > 0 else "Early"
    )
    fig3 = px.bar(
        state_delay,
        x="state",
        y="mean_delay",
        color="color",
        color_discrete_map={"Late": "#DC2626", "Early": "#16A34A"},
        labels={"mean_delay": "Mean delay (days)", "state": "State"},
    )
    fig3.add_hline(y=0, line_color="black", line_width=0.8)
    fig3.update_layout(margin=dict(t=10, b=10), showlegend=True, legend_title_text="")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Customer Forgiveness Window")
    subset = delivered[["delivery_delay", "review_score"]].dropna()
    buckets = range(-5, 21)
    records = []
    for b in buckets:
        group = subset[subset["delivery_delay"].round().astype(int) == b][
            "review_score"
        ]
        if len(group) >= 10:
            records.append(
                {"delay_days": b, "mean_review": group.mean(), "n": len(group)}
            )
    if records:
        fw = pd.DataFrame(records)
        fig4 = px.line(
            fw,
            x="delay_days",
            y="mean_review",
            markers=True,
            color_discrete_sequence=["#2563EB"],
            labels={
                "delay_days": "Delivery delay (days)",
                "mean_review": "Mean review score",
            },
        )
        fig4.add_vline(
            x=0,
            line_color="red",
            line_dash="dash",
            annotation_text="On time",
            annotation_position="top right",
        )
        fig4.add_vline(
            x=7,
            line_color="orange",
            line_dash="dot",
            annotation_text="Forgiveness threshold",
            annotation_position="top left",
        )
        fig4.update_yaxes(range=[1, 5.5])
        fig4.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)


def tab_segments(df: pd.DataFrame) -> None:
    rfm = load_rfm()
    if rfm is None:
        st.info(
            "RFM segments not yet generated. Run **Section 6** of the analysis notebook."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        counts = rfm["segment"].value_counts().reset_index()
        counts.columns = ["segment", "count"]
        fig = px.pie(
            counts,
            names="segment",
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monetary Value by Segment")
        fig2 = px.box(
            rfm,
            x="segment",
            y="monetary",
            color="segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"monetary": "Total spend (R$)"},
        )
        fig2.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("RFM Profile by Segment")
    profile = (
        rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1)
    )
    profile.columns = [
        "Avg Recency (days)",
        "Avg Frequency (orders)",
        "Avg Monetary (R$)",
    ]
    profile["N customers"] = rfm.groupby("segment").size()
    st.dataframe(
        profile.style.background_gradient(cmap="Blues", subset=["Avg Monetary (R$)"]),
        use_container_width=True,
    )

    st.subheader("Segment Scatter — Frequency vs Spend")
    fig3 = px.scatter(
        rfm.sample(min(5000, len(rfm)), random_state=42),
        x="frequency",
        y="monetary",
        color="segment",
        size="recency",
        size_max=15,
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"frequency": "Orders", "monetary": "Total spend (R$)"},
    )
    fig3.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)


def tab_predictor(df: pd.DataFrame) -> None:
    model = load_model()

    st.subheader("Live Review Risk Predictor")

    if model is None:
        st.warning(
            "Model not found at `reports/review_model.joblib`. "
            "Run **Section 7** of the analysis notebook to train and save it."
        )
        return

    st.caption("Enter hypothetical order parameters to predict bad-review probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        delay = st.slider(
            "Delivery delay (days)",
            -10,
            30,
            0,
            help="Positive = late, negative = early",
        )
        price = st.number_input("Order price (R$)", 10.0, 5000.0, 150.0)
        freight = st.number_input("Freight value (R$)", 0.0, 500.0, 20.0)

    with col2:
        installments = st.slider("Payment installments", 1, 12, 1)
        n_items = st.slider("Number of items", 1, 10, 1)
        days_deliver = st.slider("Days to deliver", 1, 60, max(delay + 7, 1))

    with col3:
        categories = sorted(
            df["product_category_name_english"].dropna().unique().tolist()
        )
        category = st.selectbox("Product category", categories)
        states = sorted(df["customer_state"].dropna().unique().tolist())
        customer_state = st.selectbox("Customer state", states)
        payment_type = st.selectbox(
            "Payment type", ["credit_card", "boleto", "voucher", "debit_card"]
        )

    if st.button("Predict Review Risk", type="primary"):
        freight_ratio = freight / (price + freight + 1e-9)
        row = pd.DataFrame(
            [
                {
                    "delivery_delay": float(delay),
                    "days_to_deliver": float(days_deliver),
                    "days_to_approve": 1.0,
                    "total_price": float(price),
                    "total_freight": float(freight),
                    "freight_ratio": freight_ratio,
                    "payment_installments": float(installments),
                    "n_items": float(n_items),
                    "product_weight_g": 500.0,
                    "product_volume_cm3": 1000.0,
                    "customer_state": customer_state,
                    "seller_state": "SP",
                    "primary_payment_type": payment_type,
                    "product_category_name_english": category,
                }
            ]
        )
        prob = model.predict_proba(row)[0][1]

        if prob > 0.5:
            band, color, emoji = "HIGH RISK", "#DC2626", "🔴"
        elif prob > 0.25:
            band, color, emoji = "MEDIUM RISK", "#F59E0B", "🟡"
        else:
            band, color, emoji = "LOW RISK", "#16A34A", "🟢"

        st.markdown(
            f"<h2 style='color:{color}'>{emoji} {band} — {prob:.1%} probability of bad review</h2>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Bad Review Probability", f"{prob:.1%}")
        c2.metric("Good Review Probability", f"{1 - prob:.1%}")

        if delay > 7:
            st.warning(
                f"⚠️ {delay} days late exceeds the forgiveness window (7 days). "
                "This is the primary driver. Reducing delay is the most effective lever."
            )


def main() -> None:
    st.title("🛒 Olist Brazilian E-Commerce Analysis")
    st.caption(
        "100k orders · 2016–2018 · 8 relational tables · "
        "[GitHub](https://github.com/binhvo9/olist-ecommerce-analysis)"
    )

    df = load_master()
    kpi_row(df)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Overview",
            "🚚 Delivery Performance",
            "👥 Customer Segments",
            "🔮 Review Predictor",
        ]
    )
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
