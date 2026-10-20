import streamlit as st
import pandas as pd
import joblib

# Load saved model and threshold
bundle = joblib.load("bank_marketing_model.pkl")
model = bundle["model"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]

st.set_page_config(
    page_title="Bank Marketing Response Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Marketing Response Prediction")
st.write(
    "Predict whether a customer is likely to subscribe "
    "to a bank term deposit."
)

st.info(
    "Educational ML project. Predictions are estimates, "
    "not guarantees."
)

st.subheader("Enter Customer Details")

with st.form("customer_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox(
        "Education", ["primary", "secondary", "tertiary", "unknown"]
    )
    default = st.selectbox("Credit in Default?", ["no", "yes"])
    balance = st.number_input("Account Balance", value=1000)
    housing = st.selectbox("Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox(
        "Last Contact Month",
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    campaign = st.number_input(
        "Contacts During This Campaign", min_value=1, value=1
    )
    pdays = st.number_input(
        "Days Since Previous Campaign Contact (-1 if never contacted)",
        min_value=-1, value=-1
    )
    previous = st.number_input(
        "Contacts Before This Campaign", min_value=0, value=0
    )
    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )

    submitted = st.form_submit_button("Predict Subscription")

if submitted:
    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    customer = customer[feature_columns]
    probability = model.predict_proba(customer)[0, 1]
    prediction = probability >= threshold

    st.subheader("Prediction Result")
    st.metric("Estimated Subscription Probability", f"{probability:.1%}")

    if prediction:
        st.success("The model predicts: Likely to subscribe")
    else:
        st.warning("The model predicts: Unlikely to subscribe")

    st.caption(f"Decision threshold: {threshold:.2f}")
