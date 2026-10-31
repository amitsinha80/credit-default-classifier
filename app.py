"""Streamlit app: evaluate pre-trained credit-default classifiers on uploaded test data."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from evaluation import compute_metrics, positive_class_scores

MODEL_DIR = Path("model")
SAMPLE_TEST_PATH = Path("test_data.csv")
TARGET = "default_next_month"
CLASS_LABELS = ["No default (0)", "Default (1)"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

st.set_page_config(page_title="Credit Default Classifier", page_icon="💳", layout="wide")


@st.cache_resource
def load_model(file_name):
    """Load a saved model once and reuse it across reruns."""
    return joblib.load(MODEL_DIR / file_name)


def evaluate(model, data):
    """Predict with a saved model and return (y_true, y_pred, metrics)."""
    X = data[list(model.feature_names_in_)]
    y_true = data[TARGET]
    y_pred = model.predict(X)
    metrics = compute_metrics(y_true, y_pred, positive_class_scores(model, X))
    return y_true, y_pred, metrics


@st.cache_data
def evaluate_all(data):
    """Metrics for every saved model on the uploaded data (cached per file)."""
    rows = []
    for name, file_name in MODEL_FILES.items():
        _, _, metrics = evaluate(load_model(file_name), data)
        rows.append({"Model": name, **metrics})
    return pd.DataFrame(rows).set_index("Model")


# ---------- Header & sidebar ----------
st.title("💳 Credit Card Default — Model Evaluation")
st.caption(
    "Five classifiers pre-trained on the UCI *Default of Credit Card Clients* dataset. "
    "Upload held-out test data to see how each model performs — no retraining happens here."
)

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))

    if SAMPLE_TEST_PATH.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=SAMPLE_TEST_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )

if uploaded is None:
    st.info("👈 Upload `test_data.csv` in the sidebar to begin (or download the sample first).")
    st.stop()

data = pd.read_csv(uploaded)
model = load_model(MODEL_FILES[model_name])

# The uploaded file must contain every training feature plus the target
missing = [c for c in list(model.feature_names_in_) + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {', '.join(missing)}")
    st.stop()

# ---------- Selected model: metric tiles ----------
y_true, y_pred, metrics = evaluate(model, data)

st.subheader(f"{model_name} — results on {len(data):,} rows")
for col, (label, value) in zip(st.columns(len(metrics)), metrics.items()):
    col.metric(label, f"{value:.3f}")

tab_cm, tab_report, tab_compare, tab_data = st.tabs(
    ["Confusion matrix", "Classification report", "Compare all models", "Data preview"]
)

# ---------- Confusion matrix ----------
with tab_cm:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=2, linecolor="white",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown(
            f"""
- Caught **{tp:,}** of **{tp + fn:,}** actual defaulters (recall {tp / (tp + fn):.1%}).
- Missed **{fn:,}** defaulters — predicted safe but did default.
- Wrongly flagged **{fp:,}** customers who did not default.
- Correctly cleared **{tn:,}** non-defaulters.
"""
        )

# ---------- Classification report ----------
with tab_report:
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_LABELS, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3))

# ---------- All models side by side ----------
with tab_compare:
    st.markdown("All five saved models evaluated on the uploaded file. **Bold** = best in each column.")
    comparison = evaluate_all(data)
    st.dataframe(
        comparison.style.format("{:.3f}").highlight_max(axis=0, props="font-weight: bold;")
    )
    st.markdown("**AUC, F1 and MCC by model** — more informative than accuracy on imbalanced data")
    st.bar_chart(comparison[["AUC", "F1", "MCC"]], stack=False)

# ---------- Data preview ----------
with tab_data:
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Default rate", f"{data[TARGET].mean():.1%}")
    st.dataframe(data.head(20))