"""
Credit Card Fraud Detection — Streamlit App
Loads the trained ANN (class-weights baseline, tuned threshold) and lets the user
either upload a CSV of transactions or enter a single transaction manually.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

st.set_page_config(page_title="Credit Card Fraud Detector (ANN)", page_icon="💳", layout="wide")

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLUMNS = ["Time", "Amount"]


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model_classweights.keras")
    scaler = joblib.load("scaler.pkl")
    config = joblib.load("model_config.pkl")
    return model, scaler, config["threshold"]


def preprocess(df: pd.DataFrame, scaler) -> np.ndarray:
    df = df.copy()
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[FEATURE_COLUMNS]
    df[SCALE_COLUMNS] = scaler.transform(df[SCALE_COLUMNS])
    return df.values


def predict(df: pd.DataFrame, model, scaler, threshold: float) -> pd.DataFrame:
    X = preprocess(df, scaler)
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)
    out = df.copy()
    out["fraud_probability"] = probs
    out["prediction"] = np.where(preds == 1, "FRAUD", "legit")
    return out


def main():
    st.title("💳 Credit Card Fraud Detection")
    st.caption(
        "ANN trained with class weights on the highly imbalanced Kaggle credit-card "
        "fraud dataset (0.17% fraud rate). Decision threshold tuned for best F1, "
        "per the project's ANN workflow documentation."
    )

    model, scaler, threshold = load_artifacts()

    with st.sidebar:
        st.header("Model info")
        st.write(f"**Decision threshold:** {threshold:.4f}")
        st.write("**Architecture:** Dense(32, relu) → Dropout(0.3) → Dense(16, relu) → Dense(1, sigmoid)")
        st.write("**Loss:** binary cross-entropy, class-weighted")
        st.write("**Why not 0.5?** With ~0.17% fraud rate, a 0.5 threshold gives many "
                 "false positives. The tuned threshold maximizes F1 on the test set.")
        threshold = st.slider("Override decision threshold", 0.0, 1.0, float(threshold), 0.001)

    tab1, tab2 = st.tabs(["📄 Batch: upload CSV", "✍️ Single transaction"])

    with tab1:
        st.subheader("Upload a CSV of transactions")
        st.write(f"Required columns: `{', '.join(FEATURE_COLUMNS)}` (same schema as the original `creditcard.csv`).")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            try:
                result = predict(df, model, scaler, threshold)
                n_fraud = (result["prediction"] == "FRAUD").sum()
                st.success(f"Scored {len(result)} transactions — flagged {n_fraud} as fraud.")
                st.dataframe(
                    result.sort_values("fraud_probability", ascending=False),
                    use_container_width=True,
                )
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", csv, "fraud_predictions.csv", "text/csv")
            except ValueError as e:
                st.error(str(e))

    with tab2:
        st.subheader("Enter a single transaction")
        st.write("Paste comma-separated values in feature order, or fill defaults (0.0) and adjust Time/Amount.")
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time (seconds since first transaction)", value=0.0)
        with col2:
            amount_val = st.number_input("Amount", value=0.0)

        raw = st.text_area(
            "V1..V28 (comma-separated, 28 values)",
            value=", ".join(["0.0"] * 28),
            height=100,
        )

        if st.button("Predict"):
            try:
                v_values = [float(x.strip()) for x in raw.split(",")]
                if len(v_values) != 28:
                    st.error(f"Expected 28 values for V1..V28, got {len(v_values)}.")
                else:
                    row = {"Time": time_val, "Amount": amount_val}
                    for i, v in enumerate(v_values, start=1):
                        row[f"V{i}"] = v
                    single_df = pd.DataFrame([row])
                    result = predict(single_df, model, scaler, threshold)
                    prob = result["fraud_probability"].iloc[0]
                    pred = result["prediction"].iloc[0]
                    if pred == "FRAUD":
                        st.error(f"⚠️ Predicted: FRAUD (probability {prob:.4f})")
                    else:
                        st.success(f"✅ Predicted: legit (probability {prob:.4f})")
            except ValueError:
                st.error("Could not parse V1..V28 — make sure it's 28 comma-separated numbers.")


if __name__ == "__main__":
    main()
