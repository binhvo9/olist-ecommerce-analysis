"""
Angle 2 – RFM Customer Segmentation
=====================================
Functions:
    build_rfm          – compute Recency / Frequency / Monetary per customer
    fit_kmeans         – scale + cluster with KMeans
    elbow_plot         – inertia vs k, optional PNG output
    label_segments     – assign human-readable segment names
    segment_profile    – summary statistics per segment
    plot_rfm_segments  – bar charts of mean R/F/M by segment
    plot_segment_scatter – scatter F vs M coloured by segment
    run_analysis       – end-to-end pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# 1. Build RFM table
# ---------------------------------------------------------------------------


def build_rfm(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Return one row per customer_unique_id with recency, frequency, monetary.

    Parameters
    ----------
    df:
        Master / cleaned DataFrame that must contain:
        ``order_status``, ``order_purchase_timestamp``,
        ``order_id``, ``total_payment``, ``customer_unique_id``.
    snapshot_date:
        Reference date for recency calculation.  Defaults to
        max(order_purchase_timestamp) + 1 day.

    Returns
    -------
    pd.DataFrame with columns [customer_unique_id, recency, frequency, monetary]
    """
    delivered = df[df["order_status"] == "delivered"].copy()

    # Ensure datetime
    delivered["order_purchase_timestamp"] = pd.to_datetime(
        delivered["order_purchase_timestamp"]
    )

    if snapshot_date is None:
        snapshot_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(
            days=1
        )

    rfm = (
        delivered.groupby("customer_unique_id")
        .agg(
            recency=(
                "order_purchase_timestamp",
                lambda x: (snapshot_date - x.max()).days,
            ),
            frequency=("order_id", "nunique"),
            monetary=("total_payment", "sum"),
        )
        .reset_index()
    )

    return rfm


# ---------------------------------------------------------------------------
# 2. KMeans clustering
# ---------------------------------------------------------------------------


def fit_kmeans(
    rfm: pd.DataFrame,
    k: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    """Scale RFM features and fit KMeans.

    Parameters
    ----------
    rfm:
        DataFrame produced by :func:`build_rfm`.
    k:
        Number of clusters.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    (rfm_with_cluster, scaler, kmeans_model)
    """
    rfm = rfm.copy()
    features = ["recency", "frequency", "monetary"]

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    rfm["cluster"] = kmeans.fit_predict(X)

    return rfm, scaler, kmeans


# ---------------------------------------------------------------------------
# 3. Elbow plot
# ---------------------------------------------------------------------------


def elbow_plot(
    rfm: pd.DataFrame,
    k_range: range = range(2, 10),
    out_dir: Optional[Path] = None,
) -> list[float]:
    """Compute KMeans inertia for each k and optionally save the elbow curve.

    Parameters
    ----------
    rfm:
        DataFrame produced by :func:`build_rfm`.
    k_range:
        Iterable of k values to test.
    out_dir:
        If provided, saves ``11_elbow.png`` into this directory.

    Returns
    -------
    List of inertia values aligned with *k_range*.
    """
    features = ["recency", "frequency", "monetary"]
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])

    inertias: list[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(k_range), inertias, marker="o", linewidth=2)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method – KMeans Inertia vs k")
        plt.tight_layout()
        fig.savefig(out_dir / "11_elbow.png", dpi=150)
        plt.close(fig)

    return inertias


# ---------------------------------------------------------------------------
# 4. Label segments
# ---------------------------------------------------------------------------

_DEFAULT_AUTO_LABELS = {
    "champions_cluster": "Champions",
    "loyal_cluster": "Loyal",
    "at_risk_cluster": "At-Risk",
    "default": "Lost",
}


def label_segments(
    rfm: pd.DataFrame,
    segment_map: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """Add a *segment* column to *rfm*.

    Parameters
    ----------
    rfm:
        DataFrame with a ``cluster`` column (from :func:`fit_kmeans`).
    segment_map:
        Optional mapping ``{cluster_int: segment_name}``.  When *None*
        segments are auto-labelled based on cluster profiles:

        * Highest monetary AND lowest recency → **Champions**
        * Highest frequency (that is not Champions) → **Loyal**
        * Highest recency AND lowest frequency → **At-Risk**
        * Remaining → **Lost**

    Returns
    -------
    rfm with added ``segment`` column.
    """
    rfm = rfm.copy()

    if segment_map is not None:
        rfm["segment"] = rfm["cluster"].map(segment_map)
        return rfm

    # Auto-labelling: build cluster profile
    profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()

    # Champions: best monetary + best (lowest) recency
    # Score = normalised monetary (max = best) - normalised recency (min = best)
    monetary_norm = (profile["monetary"] - profile["monetary"].min()) / (
        profile["monetary"].max() - profile["monetary"].min() + 1e-9
    )
    recency_norm = (profile["recency"] - profile["recency"].min()) / (
        profile["recency"].max() - profile["recency"].min() + 1e-9
    )
    champion_score = monetary_norm - recency_norm
    champions_cluster = int(champion_score.idxmax())

    remaining = profile.index[profile.index != champions_cluster]

    # Loyal: highest frequency among the rest
    loyal_cluster = int(profile.loc[remaining, "frequency"].idxmax())

    remaining2 = remaining[remaining != loyal_cluster]

    # At-Risk: highest recency + lowest frequency among the rest
    if len(remaining2) > 0:
        at_risk_score = (
            profile.loc[remaining2, "recency"] - profile.loc[remaining2, "frequency"]
        )
        at_risk_cluster = int(at_risk_score.idxmax())
    else:
        at_risk_cluster = None

    def _map_cluster(c: int) -> str:
        if c == champions_cluster:
            return "Champions"
        if c == loyal_cluster:
            return "Loyal"
        if at_risk_cluster is not None and c == at_risk_cluster:
            return "At-Risk"
        return "Lost"

    rfm["segment"] = rfm["cluster"].apply(_map_cluster)
    return rfm


# ---------------------------------------------------------------------------
# 5. Segment profile
# ---------------------------------------------------------------------------


def segment_profile(rfm: pd.DataFrame) -> pd.DataFrame:
    """Group by segment and return mean R/F/M plus customer count.

    Parameters
    ----------
    rfm:
        DataFrame with ``segment`` column.

    Returns
    -------
    Summary DataFrame indexed by segment.
    """
    profile = (
        rfm.groupby("segment")
        .agg(
            n_customers=("customer_unique_id", "count"),
            recency_mean=("recency", "mean"),
            frequency_mean=("frequency", "mean"),
            monetary_mean=("monetary", "mean"),
        )
        .reset_index()
    )
    return profile


# ---------------------------------------------------------------------------
# 6. Bar charts – R/F/M by segment
# ---------------------------------------------------------------------------


def plot_rfm_segments(rfm: pd.DataFrame, out_dir: Path) -> None:
    """Save three side-by-side bar charts (mean R, F, M per segment).

    Saved as ``12_rfm_segments.png`` inside *out_dir*.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = (
        rfm.groupby("segment")[["recency", "frequency", "monetary"]]
        .mean()
        .reset_index()
    )

    metrics = [
        ("recency", "Recency (days)", "steelblue"),
        ("frequency", "Frequency (orders)", "seagreen"),
        ("monetary", "Monetary (BRL)", "darkorange"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, label, color) in zip(axes, metrics):
        ax.bar(profile["segment"], profile[col], color=color, edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("Segment")
        ax.set_ylabel(f"Mean {label}")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("RFM Profile by Segment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "12_rfm_segments.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Scatter – Frequency vs Monetary coloured by segment
# ---------------------------------------------------------------------------


def plot_segment_scatter(rfm: pd.DataFrame, out_dir: Path) -> None:
    """Scatter plot of Frequency vs Monetary, coloured by segment.

    Point size is inversely proportional to recency (more recent = larger).
    Saved as ``13_rfm_scatter.png`` inside *out_dir*.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = rfm.copy()
    # Invert recency so recent customers appear larger
    max_rec = plot_df["recency"].max()
    plot_df["size"] = (max_rec - plot_df["recency"] + 1) / max_rec * 200 + 20

    fig, ax = plt.subplots(figsize=(9, 6))
    segments = plot_df["segment"].unique()
    palette = sns.color_palette("tab10", n_colors=len(segments))

    for seg, color in zip(segments, palette):
        mask = plot_df["segment"] == seg
        ax.scatter(
            plot_df.loc[mask, "frequency"],
            plot_df.loc[mask, "monetary"],
            s=plot_df.loc[mask, "size"],
            label=seg,
            color=color,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xlabel("Frequency (number of orders)")
    ax.set_ylabel("Monetary (total spend, BRL)")
    ax.set_title(
        "Customer Segments – Frequency vs Monetary\n(size ∝ recency, larger = more recent)"
    )
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "13_rfm_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Full pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    k: int = 4,
) -> dict:
    """Run the full RFM segmentation pipeline.

    Steps:
        1. build_rfm
        2. elbow_plot (saved to out_dir)
        3. fit_kmeans
        4. label_segments
        5. segment_profile
        6. plot_rfm_segments
        7. plot_segment_scatter

    Parameters
    ----------
    df:
        Master / cleaned DataFrame.
    out_dir:
        Directory where all PNGs are saved.
    k:
        Number of KMeans clusters.

    Returns
    -------
    dict with keys: ``rfm``, ``segment_profile``, ``kmeans``, ``scaler``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rfm = build_rfm(df)
    elbow_plot(rfm, out_dir=out_dir)
    rfm, scaler, kmeans = fit_kmeans(rfm, k=k)
    rfm = label_segments(rfm)
    profile = segment_profile(rfm)
    plot_rfm_segments(rfm, out_dir)
    plot_segment_scatter(rfm, out_dir)

    return {
        "rfm": rfm,
        "segment_profile": profile,
        "kmeans": kmeans,
        "scaler": scaler,
    }
