import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Load model ----------
bundle = joblib.load("xgb_credit_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]

st.set_page_config(page_title="Credit Eligibility Predictor", layout="centered")
st.title("Credit Eligibility Prediction (XGBoost)")
st.write("Enter client information to estimate the probability of obtaining credit.")

# ---------- Input widgets ----------
# We use the most important features + some others.
# For remaining features we will use default values.

# Demographic / situation
nombre_enfant = st.number_input(
    "Number of children (NOMBRE_ENFANT)", min_value=0, max_value=20, value=0, step=1
)

marital_status = st.selectbox(
    "Marital status code (MARITAL_STATUS)",
    options=[0, 1, 2, 3],
    help="Use the same numeric codes as in the cleaned dataset."
)

sexe_encoded = st.selectbox(
    "Sex (SEXE_encoded)",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

age_scaled = st.number_input(
    "Age (scaled) (AGE_scaled)",
    help="Use the same scaling as in training (approx. standardized age). "
         "If not sure, use a value between -3 and 3.",
    min_value=-5.0, max_value=5.0, value=0.0, step=0.1
)

residence = st.number_input(
    "Residence code (RESIDENCE)",
    min_value=0, max_value=9999, value=0, step=1,
    help="Numeric code used in the dataset."
)

# Product ownership
has_bancassurance = st.selectbox("Has bancassurance", [0, 1])
has_pack          = st.selectbox("Has pack",          [0, 1])
has_mobile        = st.selectbox("Has mobile banking", [0, 1])
has_net           = st.selectbox("Has net banking",    [0, 1])

# Bank / geography codes
banque  = st.number_input("Bank code (BANQUE)",  min_value=0, max_value=999, value=0, step=1)
agence  = st.number_input("Agency code (AGENCE)", min_value=0, max_value=999, value=0, step=1)
code_ville = st.number_input("City code (CODE_VILLE)", min_value=0, max_value=9999, value=0, step=1)
ville   = st.number_input("City group code (VILLE)", min_value=0, max_value=9999, value=0, step=1)
country = st.number_input("Country code (COUNTRY)", min_value=0, max_value=9999, value=0, step=1)

flag_prop_logement = st.selectbox("Owner of housing? (FLAG_PROPRIETAIRE_LOGEMENT)", [0, 1])
flag_etranger      = st.selectbox("Foreign resident flag (FLAG_ETRANGER_RES_MAROC)", [0, 1])

# ---------- Build feature row ----------
# Start with zeros for all features
x_dict = {f: 0 for f in FEATURES}

# Fill the ones we ask in the UI
x_dict.update({
    "NOMBRE_ENFANT": nombre_enfant,
    "has_bancassurance": has_bancassurance,
    "MARITAL_STATUS": marital_status,
    "SEXE_encoded": sexe_encoded,
    "RESIDENCE": residence,
    "AGE_scaled": age_scaled,
    "has_pack": has_pack,
    "BANQUE": banque,
    "CODE_VILLE": code_ville,
    "has_mobile": has_mobile,
    "COUNTRY": country,
    "VILLE": ville,
    "FLAG_PROPRIETAIRE_LOGEMENT": flag_prop_logement,
    "has_net": has_net,
    "AGENCE": agence,
    "FLAG_ETRANGER_RES_MAROC": flag_etranger,
})

# Convert to DataFrame with correct column order
X_input = pd.DataFrame([x_dict], columns=FEATURES)

# ---------- Prediction ----------
if st.button("Predict credit decision"):
    proba = model.predict_proba(X_input)[:, 1][0]
    decision = int(proba >= THRESHOLD)

    st.subheader("Results")
    st.write(f"**Probability of obtaining credit:** {proba:.3f}")

    if decision == 1:
        st.success("Predicted decision: **CREDIT GRANTED (1)**")
    else:
        st.error("Predicted decision: **CREDIT NOT GRANTED (0)**")

    st.caption(f"Decision threshold used: {THRESHOLD:.3f}")
