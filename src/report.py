"""Executive PDF report generator using reportlab."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

W, H = A4
BRAND_BLUE = colors.HexColor("#2563EB")
BRAND_DARK = colors.HexColor("#1E293B")
LIGHT_GREY = colors.HexColor("#F1F5F9")


def _styles():
    s = getSampleStyleSheet()
    s.add(
        ParagraphStyle(
            "H1Report", parent=s["Heading1"], textColor=BRAND_BLUE, fontSize=22
        )
    )
    s.add(
        ParagraphStyle(
            "H2Report", parent=s["Heading2"], textColor=BRAND_DARK, fontSize=14
        )
    )
    s.add(ParagraphStyle("BodyReport", parent=s["BodyText"], fontSize=10, leading=14))
    s.add(
        ParagraphStyle(
            "Caption",
            parent=s["BodyText"],
            fontSize=8,
            textColor=colors.grey,
            alignment=1,
        )
    )
    return s


def _kpi_table(kpis: dict) -> Table:
    data = [["Metric", "Value"]] + [[k, str(v)] for k, v in kpis.items()]
    t = Table(data, colWidths=[10 * cm, 6 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _fig(path: Path, width: float = 14 * cm, max_height: float = 18 * cm) -> Image:
    if path.exists():
        img = Image(str(path))
        aspect = img.imageHeight / img.imageWidth
        h = width * aspect
        if h > max_height:
            h = max_height
            width = h / aspect
        img.drawWidth = width
        img.drawHeight = h
        return img
    return Paragraph(f"[Figure not found: {path.name}]", _styles()["Caption"])


def generate(
    delivery_kpis: dict,
    segment_profile,
    rf_auc: float,
    lr_auc: float,
    figures_dir: Path,
    out_path: Path,
) -> Path:
    """Generate a 5-page executive PDF report.

    Parameters
    ----------
    delivery_kpis:
        Dict from delivery.delivery_kpis()
    segment_profile:
        DataFrame from segmentation.segment_profile()
    rf_auc, lr_auc:
        ROC-AUC scores from prediction.train()
    figures_dir:
        Path to reports/figures/ containing PNG charts
    out_path:
        Destination PDF path

    Returns
    -------
    Path to generated PDF
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir)
    s = _styles()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # ── Page 1: Title & overview ──────────────────────────────────────────────
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("Olist Brazilian E-Commerce", s["H1Report"]))
    story.append(
        Paragraph("End-to-End Data Analysis — Executive Summary", s["H2Report"])
    )
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "This report presents findings from a comprehensive analysis of ~100,000 orders "
            "on the Olist marketplace (2016–2018), covering delivery performance, customer "
            "segmentation, and review score prediction.",
            s["BodyReport"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    headline_kpis = {
        "Total delivered orders": f"{delivery_kpis.get('total_delivered', 'N/A'):,}",
        "On-time delivery rate": f"{delivery_kpis.get('on_time_rate', 0):.1%}",
        "Mean delivery delay": f"{delivery_kpis.get('mean_delay_days', 0):.1f} days",
        "% very late (>7 days)": f"{delivery_kpis.get('pct_very_late', 0):.1%}",
        "Review prediction ROC-AUC (RF)": f"{rf_auc:.3f}",
        "Review prediction ROC-AUC (LR baseline)": f"{lr_auc:.3f}",
    }
    story.append(_kpi_table(headline_kpis))
    story.append(PageBreak())

    # ── Page 2: Delivery Performance ─────────────────────────────────────────
    story.append(Paragraph("Angle 1 — Delivery Performance", s["H1Report"]))
    story.append(
        Paragraph(
            "Delivery delay is defined as actual delivery date minus estimated delivery date. "
            "Negative values mean early delivery; positive values mean late. "
            f"Only {delivery_kpis.get('on_time_rate', 0):.1%} of orders arrive on or before "
            "the estimated date.",
            s["BodyReport"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(_fig(figures_dir / "07_delay_distribution.png"))
    story.append(Paragraph("Figure 1 — Delivery delay distribution", s["Caption"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_fig(figures_dir / "08_delay_by_state.png"))
    story.append(
        Paragraph("Figure 2 — Mean delay by customer state (red = late)", s["Caption"])
    )
    story.append(PageBreak())

    # ── Page 3: Forgiveness window & review impact ────────────────────────────
    story.append(Paragraph("The Customer Forgiveness Window", s["H1Report"]))
    story.append(
        Paragraph(
            "Review scores remain relatively stable for orders arriving up to 7 days late. "
            "Beyond that threshold, mean review scores drop sharply — a key operational "
            "insight for setting customer expectations and prioritising logistics investment.",
            s["BodyReport"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(_fig(figures_dir / "10_forgiveness_window.png"))
    story.append(
        Paragraph("Figure 3 — Mean review score by delivery delay bucket", s["Caption"])
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(_fig(figures_dir / "09_delay_vs_review.png"))
    story.append(
        Paragraph("Figure 4 — Mean delivery delay per review score", s["Caption"])
    )
    story.append(PageBreak())

    # ── Page 4: Customer Segmentation ────────────────────────────────────────
    story.append(Paragraph("Angle 2 — Customer Segmentation (RFM)", s["H1Report"]))
    story.append(
        Paragraph(
            "Customers were segmented using Recency, Frequency, and Monetary value (RFM) "
            "with KMeans clustering (k=4). Each segment requires a distinct business response.",
            s["BodyReport"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    if segment_profile is not None:
        seg_data = [["Segment"] + list(segment_profile.columns)]
        for idx, row in segment_profile.iterrows():
            seg_data.append(
                [str(idx)] + [str(round(v, 1)) if isinstance(v, (int, float)) else str(v) for v in row]
            )
        seg_table = Table(seg_data, colWidths=[4 * cm] * len(seg_data[0]))
        seg_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(seg_table)
        story.append(Paragraph("Table 1 — Mean RFM profile per segment", s["Caption"]))
        story.append(Spacer(1, 0.3 * cm))

    story.append(_fig(figures_dir / "12_rfm_segments.png"))
    story.append(Paragraph("Figure 5 — RFM metrics by segment", s["Caption"]))
    story.append(PageBreak())

    # ── Page 5: Prediction & Recommendations ─────────────────────────────────
    story.append(Paragraph("Angle 3 — Review Score Prediction", s["H1Report"]))
    story.append(
        Paragraph(
            f"A Random Forest classifier achieves ROC-AUC = {rf_auc:.3f} at predicting "
            f"bad reviews (scores 1–2) vs good reviews (4–5). The LR baseline scores "
            f"{lr_auc:.3f}. SHAP analysis identifies delivery_delay as the dominant "
            "feature, followed by freight_ratio and product_category.",
            s["BodyReport"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(_fig(figures_dir / "15_shap_summary.png", width=11 * cm))
    story.append(
        Paragraph("Figure 6 — SHAP feature importance (Random Forest)", s["Caption"])
    )
    story.append(PageBreak())

    story.append(Paragraph("Recommendations", s["H2Report"]))
    recommendations = [
        ["Priority", "Recommendation", "Expected Impact"],
        [
            "1 — HIGH",
            "Reduce delivery delays in North/Northeast via regional fulfilment centres",
            "Lift on-time rate from 60% → 75%+; shift ~15k orders/year away from bad-review risk",
        ],
        [
            "2 — HIGH",
            "Set customer expectations proactively when delay > 5 days is predicted",
            "Reduce score-1 reviews by softening surprise; same delay, better perception",
        ],
        [
            "3 — MEDIUM",
            "Launch re-engagement campaign for At-Risk segment (high recency, low frequency)",
            "Recover 10–20% of at-risk customers before they churn permanently",
        ],
        [
            "4 — MEDIUM",
            "Deploy review-risk score in seller dashboard; alert sellers pre-delivery",
            "Early intervention on high-risk orders reduces escalations",
        ],
    ]
    rec_table = Table(recommendations, colWidths=[3 * cm, 7 * cm, 6 * cm])
    rec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("WORDWRAP", (0, 0), (-1, -1), True),
            ]
        )
    )
    story.append(rec_table)

    doc.build(story)
    print(f"Executive PDF → {out_path}")
    return out_path
