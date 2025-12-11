import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ================== CONFIG & STYLE ==================

# Chaabi orange color (approx)
CHAABI_ORANGE = "#f58220"

st.set_page_config(
    page_title="Chaabi Credit Eligibility",
    page_icon="🐎",
    layout="centered"
)

# Inject some simple CSS for branding
st.markdown(
    f"""
    <style>
    .main {{
        background-color: #0e1117;
        color: white;
    }}
    .stButton>button {{
        background-color: {CHAABI_ORANGE};
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }}
    .stButton>button:hover {{
        background-color: #ff9b3d;
        color: white;
    }}
    .big-title {{
        font-size: 2.6rem;
        font-weight: 800;
        color: {CHAABI_ORANGE};
    }}
    .subtitle {{
        font-size: 1rem;
        color: #d0d0d0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== LOAD MODEL BUNDLE ==================

bundle = joblib.load("xgb_credit_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = float(bundle["threshold"])

# ================== HEADER ==================

# Logo (put chaabi_logo.png in the same folder)
st.image("chaabi_logo.png", width=140)

st.markdown('<div class="big-title">Credit Eligibility Prediction</div>', unsafe_allow_html=True)
st.write("")
st.markdown(
    '<div class="subtitle">'
    "Interactive tool to estimate the probability that a client will obtain credit "
    "based on their profile and existing products."
    "</div>",
    unsafe_allow_html=True,
)
st.write("---")

# ================== MAPPINGS (ADJUST IF NEEDED) ==================

# NOTE: Make sure these codes match the ones used during training.
MARITAL_STATUS_MAP = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2,
    "Widowed": 3,
}

SEX_MAP = {
    "Female": 0,
    "Male": 1,
}

YES_NO_MAP = {
    "No": 0,
    "Yes": 1,
}

# If you know the mean/std used to create AGE_scaled, you can use real age and rescale.
# Example (replace with your actual numbers if available):
AGE_MEAN = 40.0   # <-- TODO: put your real training mean if you know it
AGE_STD = 10.0    # <-- TODO: put your real training std if you know it

# ================== INPUT FORM ==================

st.subheader("Client profile")

col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
    )

    nombre_enfant = st.number_input(
        "Number of children",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    marital_status_label = st.selectbox(
        "Marital status",
        options=list(MARITAL_STATUS_MAP.keys()),
    )

    sex_label = st.selectbox(
        "Sex",
        options=list(SEX_MAP.keys()),
    )

    residence_code = st.number_input(
        "Residence code",
        min_value=0,
        max_value=9999,
        value=0,
        step=1,
        help="Internal residence code used by the bank.",
    )

with col2:
    has_bancassurance_label = st.selectbox(
        "Has bancassurance product?",
        options=list(YES_NO_MAP.keys()),
    )
    has_pack_label = st.selectbox(
        "Has banking pack?",
        options=list(YES_NO_MAP.keys()),
    )
    has_mobile_label = st.selectbox(
        "Uses mobile banking?",
        options=list(YES_NO_MAP.keys()),
    )
    has_net_label = st.selectbox(
        "Uses online banking (web)?",
        options=list(YES_NO_MAP.keys()),
    )

st.subheader("Bank & location information")

col3, col4 = st.columns(2)

with col3:
    banque = st.number_input(
        "Bank code",
        min_value=0, max_value=999, value=0, step=1,
        help="Internal code of the bank entity."
    )
    agence = st.number_input(
        "Branch / agency code",
        min_value=0, max_value=999, value=0, step=1,
        help="Internal code of the local branch."
    )
    flag_owner_label = st.selectbox(
        "Client owns their housing?",
        options=list(YES_NO_MAP.keys()),
    )

with col4:
    code_ville = st.number_input(
        "City code",
        min_value=0, max_value=9999, value=0, step=1,
        help="Internal city code."
    )
    ville = st.number_input(
        "City group / zone code",
        min_value=0, max_value=9999, value=0, step=1,
    )
    country = st.number_input(
        "Country code",
        min_value=0, max_value=9999, value=0, step=1,
    )
    flag_etranger_label = st.selectbox(
        "Client is a foreign resident?",
        options=list(YES_NO_MAP.keys()),
    )

st.write("---")

# ================== BUILD FEATURE ROW ==================

# Start with zeros for all features
x_dict = {f: 0 for f in FEATURES}

# Convert nice labels to encoded values
marital_status = MARITAL_STATUS_MAP[marital_status_label]
sexe_encoded = SEX_MAP[sex_label]
has_bancassurance = YES_NO_MAP[has_bancassurance_label]
has_pack = YES_NO_MAP[has_pack_label]
has_mobile = YES_NO_MAP[has_mobile_label]
has_net = YES_NO_MAP[has_net_label]
flag_prop_logement = YES_NO_MAP[flag_owner_label]
flag_etranger = YES_NO_MAP[flag_etranger_label]

# Age scaling (if AGE_MEAN/STD not exact, app is still ok for demo,
# but you can plug the true values from your preprocessing).
age_scaled = (age_years - AGE_MEAN) / AGE_STD

x_dict.update({
    "NOMBRE_ENFANT": nombre_enfant,
    "has_bancassurance": has_bancassurance,
    "MARITAL_STATUS": marital_status,
    "SEXE_encoded": sexe_encoded,
    "RESIDENCE": residence_code,
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

X_input = pd.DataFrame([x_dict], columns=FEATURES)

# ================== PREDICTION ==================

st.markdown("### Prediction")

if st.button("Predict credit decision"):
    proba = model.predict_proba(X_input)[:, 1][0]
    decision = int(proba >= THRESHOLD)

    st.write("")
    st.metric("Probability of obtaining credit", f"{proba:.1%}")
    st.progress(int(proba * 100))

    st.write("")
    if decision == 1:
        st.success("**Predicted decision: CREDIT GRANTED**")
    else:
        st.error("**Predicted decision: CREDIT NOT GRANTED**")

    st.caption(f"Internal decision threshold used: {THRESHOLD:.3f}")
else:
    st.info("Fill in the client information and click **Predict credit decision**.")
