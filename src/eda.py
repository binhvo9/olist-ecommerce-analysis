from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

sns.set_theme(style="whitegrid")


def plot_monthly_orders(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly = (
        df["order_purchase_timestamp"]
        .dropna()
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=1.8)
    ax.set_title("Monthly Order Volume")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Orders")
    fig.tight_layout()
    fig.savefig(out_dir / "01_monthly_orders.png", dpi=150)
    plt.close(fig)


def plot_orders_by_state(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_counts = (
        df["customer_state"].value_counts().head(15).sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=state_counts.index, y=state_counts.values, ax=ax)
    ax.set_title("Top 15 States by Order Count")
    ax.set_xlabel("State")
    ax.set_ylabel("Number of Orders")
    fig.tight_layout()
    fig.savefig(out_dir / "02_orders_by_state.png", dpi=150)
    plt.close(fig)


def plot_revenue_by_category(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_revenue = (
        df.groupby("product_category_name_english")["total_price"]
        .sum()
        .dropna()
        .nlargest(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    cat_revenue.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Categories by Revenue (BRL)")
    ax.set_xlabel("Total Revenue (BRL)")
    ax.set_ylabel("Category")
    fig.tight_layout()
    fig.savefig(out_dir / "03_revenue_by_category.png", dpi=150)
    plt.close(fig)


def plot_review_score_dist(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_counts = df["review_score"].dropna().astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=score_counts.index, y=score_counts.values, ax=ax)
    ax.set_title("Review Score Distribution (1–5)")
    ax.set_xlabel("Review Score")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "04_review_scores.png", dpi=150)
    plt.close(fig)


def plot_payment_installments(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    install_counts = (
        df["payment_installments"].dropna().astype(int).value_counts().sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x=install_counts.index, y=install_counts.values, ax=ax)
    ax.set_title("Payment Installment Distribution")
    ax.set_xlabel("Number of Installments")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "05_installments.png", dpi=150)
    plt.close(fig)


def plot_payment_types(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    type_counts = df["primary_payment_type"].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.set_title("Payment Type Breakdown")
    fig.tight_layout()
    fig.savefig(out_dir / "06_payment_types.png", dpi=150)
    plt.close(fig)


def generate_profile_report(df: pd.DataFrame, out_path: Path) -> None:
    from ydata_profiling import ProfileReport

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = ProfileReport(df, minimal=True, title="Olist Master Dataset Profile")
    profile.to_file(out_path)


def run_all(df: pd.DataFrame, out_dir: Path, reports_dir: Path) -> None:
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)

    plot_monthly_orders(df, out_dir)
    plot_orders_by_state(df, out_dir)
    plot_revenue_by_category(df, out_dir)
    plot_review_score_dist(df, out_dir)
    plot_payment_installments(df, out_dir)
    plot_payment_types(df, out_dir)
    generate_profile_report(df, reports_dir / "profile_report.html")
