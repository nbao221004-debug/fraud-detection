"""Streamlit web app: fraud detection for financial transactions.

Run: streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import fraud_features  # noqa: F401  (needed so joblib can load the add_features function)
from fraud_features import RAW_COLUMNS

BASE = Path(__file__).parent
st.set_page_config(page_title="Fraud detection", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_model():
    pipe = joblib.load(BASE / "model" / "fraud_pipeline.joblib")
    meta = json.loads((BASE / "model" / "metadata.json").read_text(encoding="utf-8"))
    return pipe, meta


pipe, meta = load_model()

EXAMPLES = {
    "Suspicious transaction (account emptied)": dict(type="TRANSFER", amount=181.0, oldbalanceOrg=181.0,
                                                     newbalanceOrig=0.0, oldbalanceDest=0.0, newbalanceDest=0.0),
    "Normal payment": dict(type="PAYMENT", amount=9839.64, oldbalanceOrg=170136.0,
                           newbalanceOrig=160296.36, oldbalanceDest=0.0, newbalanceDest=0.0),
    "Normal cash-out": dict(type="CASH_OUT", amount=229133.94, oldbalanceOrg=15325.0,
                            newbalanceOrig=0.0, oldbalanceDest=5083.0, newbalanceDest=51513.44),
}

st.title("🛡️ Fraud detection")
st.caption(f"Model: {meta['model_name']} · Capstone project, simulated mobile-money transactions")

with st.sidebar:
    st.header("Settings")
    st.write("**Decision threshold**: a transaction whose fraud probability is at or above the threshold raises an alert.")
    mode = st.radio("Mode", ["Balanced (F1-optimal)", "Custom"], index=0)
    if mode.startswith("Balanced"):
        threshold = meta["threshold_f1"]
        st.metric("Threshold", f"{threshold:.2f}")
    else:
        threshold = st.slider("Threshold", 0.01, 0.99, float(meta["threshold_f1"]), 0.01)
    st.info("Lower threshold: catches more fraud but raises more false alarms.")

tab1, tab2 = st.tabs(["Single transaction", "Upload CSV"])

with tab1:
    choice = st.selectbox("Example", ["(enter manually)"] + list(EXAMPLES))
    ex = EXAMPLES.get(choice, dict(type="TRANSFER", amount=1000.0, oldbalanceOrg=5000.0,
                                   newbalanceOrig=4000.0, oldbalanceDest=0.0, newbalanceDest=1000.0))
    types = meta["types"]
    with st.form("single"):
        t = st.selectbox("Transaction type", types, index=types.index(ex["type"]))
        c1, c2 = st.columns(2)
        amount = c1.number_input("Amount", min_value=0.0, value=float(ex["amount"]), step=100.0)
        oldo = c1.number_input("Sender balance before (oldbalanceOrg)", min_value=0.0, value=float(ex["oldbalanceOrg"]), step=100.0)
        newo = c1.number_input("Sender balance after (newbalanceOrig)", min_value=0.0, value=float(ex["newbalanceOrig"]), step=100.0)
        oldd = c2.number_input("Recipient balance before (oldbalanceDest)", min_value=0.0, value=float(ex["oldbalanceDest"]), step=100.0)
        newd = c2.number_input("Recipient balance after (newbalanceDest)", min_value=0.0, value=float(ex["newbalanceDest"]), step=100.0)
        go = st.form_submit_button("Check transaction", type="primary")
    if go:
        row = pd.DataFrame([dict(type=t, amount=amount, oldbalanceOrg=oldo, newbalanceOrig=newo,
                                 oldbalanceDest=oldd, newbalanceDest=newd)])[RAW_COLUMNS]
        p = float(pipe.predict_proba(row)[0, 1])
        st.progress(min(p, 1.0), text=f"Fraud probability: {p:.1%}")
        if p >= threshold:
            st.error("ALERT: this transaction shows signs of fraud")
        else:
            st.success("OK: this transaction looks normal")

with tab2:
    st.write("The CSV file must contain the columns: " + ", ".join(RAW_COLUMNS))
    up = st.file_uploader("Choose a CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up, encoding="utf-8-sig")
        missing = [c for c in RAW_COLUMNS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            proba = pipe.predict_proba(data[RAW_COLUMNS])[:, 1]
            out = data.copy()
            out["fraud_probability"] = proba
            out["alert"] = (proba >= threshold).astype(int)
            st.write(f"Scored {len(out):,} transactions; {int(out['alert'].sum()):,} raised an alert.")
            st.dataframe(out.sort_values("fraud_probability", ascending=False).head(100))
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")

st.divider()
st.caption("Note: the model was trained on simulated data for a university project and must not be used in a real banking system.")
