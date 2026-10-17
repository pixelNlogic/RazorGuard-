
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------ CONFIGURATION ------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="RazorGuard | Fraud Risk Manager",
    page_icon="🛡️",
    layout="wide",
)

# ------------------ STYLING ------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220, #111c33);
        color: #f1f5f9;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    h1, h2, h3 {
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ LOAD MODEL ------------------

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        "Model files are missing. Run `python train_model.py` first."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

except Exception as error:
    st.error(f"Could not load the model or metrics: {error}")
    st.stop()

# ------------------ HEADER ------------------

st.title("🛡️ RazorGuard")
st.subheader("AI-Powered Fraud Risk Manager")
st.caption(
    "Machine learning for transaction fraud detection | "
    "Educational prototype using an anonymized public dataset"
)

st.divider()

# ------------------ METRICS ------------------

st.header("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Precision",
    f"{metrics.get('precision', 0):.2%}"
)

c2.metric(
    "Recall",
    f"{metrics.get('recall', 0):.2%}"
)

c3.metric(
    "F1 Score",
    f"{metrics.get('f1_score', 0):.4f}"
)

c4.metric(
    "Average Precision",
    f"{metrics.get('average_precision', 0):.4f}"
)

st.caption(
    "Metrics are from the held-out test set. "
    "Performance on future transactions may differ."
)

st.divider()

# ------------------ CONFUSION MATRIX ------------------

st.header("Test Set Evaluation")

tn = int(metrics.get("true_negatives", 0))
fp = int(metrics.get("false_positives", 0))
fn = int(metrics.get("false_negatives", 0))
tp = int(metrics.get("true_positives", 0))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Correct legitimate", f"{tn:,}")
m2.metric("False alarms", f"{fp:,}")
m3.metric("Missed fraud", f"{fn:,}")
m4.metric("Fraud detected", f"{tp:,}")

matrix = pd.DataFrame(
    [
        [tn, fp],
        [fn, tp],
    ],
    index=["Actual legitimate", "Actual fraud"],
    columns=["Predicted legitimate", "Predicted fraud"],
)

st.write("**Confusion matrix**")
st.dataframe(matrix, use_container_width=True)

st.info(
    "A false alarm flags a legitimate transaction. "
    "A missed fraud is a fraudulent transaction classified as legitimate."
)

st.divider()

# ------------------ TRANSACTION PREDICTION ------------------

st.header(" Transaction Risk Checker")

st.write(
    "Enter the dataset features to estimate a transaction's fraud risk. "
    "This demo uses anonymized V1–V28 features, not real-world "
    "customer or device information."
)

threshold = st.slider(
    "Fraud decision threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(metrics.get("threshold", 0.95)),
    step=0.01,
    help=(
        "Lower thresholds flag more transactions and may increase false "
        "alarms. Higher thresholds flag fewer transactions but may miss "
        "more fraud."
    ),
)

with st.form("transaction_form"):
    st.subheader("Transaction details")

    amount = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    time_value = st.number_input(
        "Time feature",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Elapsed seconds from the dataset's reference point.",
    )

    st.write("Anonymized transaction features")

    features = {}

    # Use two columns so the form remains compact.
    left, right = st.columns(2)

    for i in range(1, 29):
        target_column = left if i <= 14 else right

        with target_column:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                step=0.1,
                format="%.4f",
                key=f"feature_V{i}",
            )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        use_container_width=True,
    )

if submitted:
    input_data = {
        "Time": time_value,
        **features,
        "Amount": amount,
    }

    try:
        model_input = pd.DataFrame([input_data])

        # Preserve the feature order used by the trained model.
        if hasattr(model, "feature_names_in_"):
            model_input = model_input[
                list(model.feature_names_in_)
            ]

        fraud_probability = float(
            model.predict_proba(model_input)[0][1]
        )

        st.subheader("Risk Assessment")

        r1, r2 = st.columns(2)

        r1.metric(
            "Estimated fraud probability",
            f"{fraud_probability:.2%}",
        )

        r2.metric(
            "Decision threshold",
            f"{threshold:.2%}",
        )

        if fraud_probability >= threshold:
            st.error(
                " HIGH RISK: Flagged by the model for review."
            )
        else:
            st.success(
                " LOWER RISK: Not flagged at this threshold."
            )

        st.caption(
            "This is a model score, not proof of fraud. "
            "The anonymized dataset features may not represent a "
            "real transaction's context."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "RazorGuard | Student project | "
    "Not a production payment-security system"
)
