"""Angle 3 — Review Score Prediction (binary: bad vs good review)."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "delivery_delay",
    "days_to_deliver",
    "days_to_approve",
    "total_price",
    "total_freight",
    "freight_ratio",
    "payment_installments",
    "n_items",
    "product_weight_g",
    "product_volume_cm3",
]
CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "primary_payment_type",
    "product_category_name_english",
]


def build_model_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to delivered orders, drop neutral reviews (score==3), add binary target."""
    delivered = df[df["order_status"] == "delivered"].copy()
    model_df = delivered[delivered["review_score"] != 3].copy()
    model_df["bad_review"] = (model_df["review_score"] <= 2).astype(int)
    return model_df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train baseline (LR) and final (RF) pipelines. Returns evaluation metrics + fitted RF."""
    model_df = build_model_df(df)
    X = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_df["bad_review"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()

    lr = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=500, random_state=random_state)),
        ]
    )
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    lr_report = classification_report(y_test, lr.predict(X_test), output_dict=True)

    rf = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    rf_report = classification_report(y_test, rf.predict(X_test), output_dict=True)

    return {
        "rf_pipeline": rf,
        "lr_pipeline": lr,
        "X_test": X_test,
        "y_test": y_test,
        "lr_auc": lr_auc,
        "rf_auc": rf_auc,
        "lr_report": lr_report,
        "rf_report": rf_report,
        "class_balance": float(y.mean()),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def get_shap_values(rf_pipeline: Pipeline, X_sample: pd.DataFrame):
    """Compute SHAP values for the RF classifier on a sample."""
    import shap

    X_transformed = rf_pipeline["prep"].transform(X_sample)
    feature_names = NUMERIC_FEATURES + list(
        rf_pipeline["prep"]
        .named_transformers_["cat"]["encode"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    explainer = shap.TreeExplainer(rf_pipeline["clf"])
    shap_values = explainer.shap_values(X_transformed)
    return shap_values, X_transformed, feature_names


def plot_confusion_matrix(rf_pipeline: Pipeline, X_test, y_test, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        rf_pipeline, X_test, y_test, ax=ax, display_labels=["Good", "Bad"]
    )
    ax.set_title("Random Forest — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "14_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_shap_summary(rf_pipeline: Pipeline, X_sample, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import shap

    shap_values, X_transformed, feature_names = get_shap_values(rf_pipeline, X_sample)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # shap_values may be list (multi-class) or 2D array (binary)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(
        vals,
        X_transformed,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "15_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_model(rf_pipeline: Pipeline, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "review_model.joblib"
    joblib.dump(rf_pipeline, path)
    return path


def run_analysis(df: pd.DataFrame, out_dir: Path, reports_dir: Path) -> dict:
    """Full Angle 3 pipeline: train → evaluate → SHAP → save model."""
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)

    results = train(df)
    rf = results["rf_pipeline"]

    print(f"LR  ROC-AUC: {results['lr_auc']:.3f}")
    print(f"RF  ROC-AUC: {results['rf_auc']:.3f}")
    print(f"Class balance (bad review rate): {results['class_balance']:.1%}")

    plot_confusion_matrix(rf, results["X_test"], results["y_test"], out_dir)

    shap_sample = results["X_test"].sample(
        min(500, len(results["X_test"])), random_state=42
    )
    plot_shap_summary(rf, shap_sample, out_dir)

    model_path = save_model(rf, reports_dir)
    print(f"Model saved → {model_path}")

    return results
