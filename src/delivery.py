"""
Angle 1 — Delivery Performance Analysis
========================================
Functions for computing delivery KPIs, state-level breakdowns, and the
relationship between delivery delay and review scores.

All plotting functions accept an already-cleaned master DataFrame (output of
clean.clean()) and write figures to *out_dir*.  filter_delivered() is called
internally where the analysis scope is restricted to delivered orders.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from src.clean import filter_delivered

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


def delivery_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of top-level delivery KPIs.

    Parameters
    ----------
    df:
        Cleaned master DataFrame (all statuses).  filter_delivered() is applied
        internally.

    Returns
    -------
    dict with keys:
        on_time_rate, median_delay_days, mean_delay_days,
        pct_very_late, total_delivered
    """
    delivered = filter_delivered(df)
    delay = delivered["delivery_delay"].dropna()

    on_time_rate = (
        float((delay <= 0).sum() / len(delay)) if len(delay) > 0 else float("nan")
    )
    median_delay_days = float(delay.median())
    mean_delay_days = float(delay.mean())
    pct_very_late = (
        float((delay > 7).sum() / len(delay)) if len(delay) > 0 else float("nan")
    )
    total_delivered = int(len(delivered))

    return {
        "on_time_rate": on_time_rate,
        "median_delay_days": median_delay_days,
        "mean_delay_days": mean_delay_days,
        "pct_very_late": pct_very_late,
        "total_delivered": total_delivered,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_delay_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of delivery_delay clipped to [-20, 60] days.

    A red dashed vertical line marks 0 (on-time boundary).
    Saved as 07_delay_distribution.png.
    """
    delivered = filter_delivered(df)
    delay = delivered["delivery_delay"].dropna().clip(-20, 60)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(delay, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(
        0, color="red", linestyle="--", linewidth=1.5, label="On-time threshold (0)"
    )
    ax.set_xlabel("Delivery Delay (days, clipped to [-20, 60])")
    ax.set_ylabel("Number of Orders")
    ax.set_title("Distribution of Delivery Delay\n(positive = late, negative = early)")
    ax.legend()
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "07_delay_distribution.png", dpi=150)
    plt.close(fig)


def plot_delay_by_state(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of mean delivery_delay per customer_state.

    Bars are coloured red (mean_delay > 0, late) or green (mean_delay <= 0,
    early/on-time), sorted descending by mean delay.
    Saved as 08_delay_by_state.png.
    """
    delivered = filter_delivered(df)
    state_delay = (
        delivered.groupby("customer_state")["delivery_delay"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    state_delay.columns = ["customer_state", "mean_delay"]

    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in state_delay["mean_delay"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(state_delay["customer_state"], state_delay["mean_delay"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Customer State")
    ax.set_ylabel("Mean Delivery Delay (days)")
    ax.set_title(
        "Mean Delivery Delay by Customer State\n(red = late on average, green = early)"
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "08_delay_by_state.png", dpi=150)
    plt.close(fig)


def plot_delay_vs_review(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart of mean delivery_delay per review_score (1–5).

    Saved as 09_delay_vs_review.png.
    """
    delivered = filter_delivered(df)
    review_delay = (
        delivered.groupby("review_score")["delivery_delay"]
        .mean()
        .reindex([1, 2, 3, 4, 5])
        .reset_index()
    )
    review_delay.columns = ["review_score", "mean_delay"]

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("RdYlGn", n_colors=5)
    ax.bar(
        review_delay["review_score"].astype(str),
        review_delay["mean_delay"],
        color=palette,
        edgecolor="white",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Review Score")
    ax.set_ylabel("Mean Delivery Delay (days)")
    ax.set_title("Mean Delivery Delay by Review Score")
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "09_delay_vs_review.png", dpi=150)
    plt.close(fig)


def plot_forgiveness_window(df: pd.DataFrame, out_dir: Path) -> None:
    """Line chart: mean review score for each integer delay bucket from -5 to +20.

    Highlights the threshold where review score drops sharply.
    Saved as 10_forgiveness_window.png.
    """
    delivered = filter_delivered(df)
    subset = delivered[["delivery_delay", "review_score"]].dropna()

    buckets = range(-5, 21)
    records = []
    for bucket in buckets:
        mask = subset["delivery_delay"].round().astype(int) == bucket
        group = subset.loc[mask, "review_score"]
        if len(group) > 0:
            records.append({"delay_days": bucket, "mean_review_score": group.mean()})

    bucket_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        bucket_df["delay_days"],
        bucket_df["mean_review_score"],
        marker="o",
        linewidth=2,
        color="steelblue",
        markersize=5,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="On-time boundary")
    ax.set_xlabel("Delivery Delay (days, integer bucket)")
    ax.set_ylabel("Mean Review Score")
    ax.set_title(
        "Customer Forgiveness Window\n(mean review score by delivery delay bucket)"
    )
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.set_ylim(1, 5.5)
    ax.legend()
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "10_forgiveness_window.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# State performance table
# ---------------------------------------------------------------------------


def state_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with per-state delivery statistics.

    Returns
    -------
    pd.DataFrame with columns:
        customer_state, n_orders, on_time_rate, mean_delay, median_delay
    Sorted by mean_delay descending.
    """
    delivered = filter_delivered(df)

    def _on_time(s):
        s = s.dropna()
        return float((s <= 0).sum() / len(s)) if len(s) > 0 else float("nan")

    agg = (
        delivered.groupby("customer_state")["delivery_delay"]
        .agg(
            n_orders="count",
            mean_delay="mean",
            median_delay="median",
            on_time_rate=_on_time,
        )
        .reset_index()
    )

    agg = agg[
        ["customer_state", "n_orders", "on_time_rate", "mean_delay", "median_delay"]
    ]
    agg = agg.sort_values("mean_delay", ascending=False).reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_analysis(df: pd.DataFrame, out_dir: Path) -> dict:
    """Run the full Angle 1 delivery performance analysis.

    Parameters
    ----------
    df:
        Cleaned master DataFrame.
    out_dir:
        Directory where PNG figures are saved.

    Returns
    -------
    dict containing all KPI values plus 'state_table' (pd.DataFrame).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kpis = delivery_kpis(df)

    plot_delay_distribution(df, out_dir)
    plot_delay_by_state(df, out_dir)
    plot_delay_vs_review(df, out_dir)
    plot_forgiveness_window(df, out_dir)

    state_table = state_performance_table(df)

    return {**kpis, "state_table": state_table}
