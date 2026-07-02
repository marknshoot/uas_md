from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


pipeline_path = Path("models/pipeline.joblib")


@st.cache_resource
def load_pipeline():
    return joblib.load(pipeline_path)


try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(f"Could not load model: {e}")
    st.error(
        f"Make sure {pipeline_path} exists. "
        f"Run `python main.py` first."
    )
    st.stop()


label_map = {0: "good", 1: "poor", 2: "standard"}

feature_names = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
    "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
    "Monthly_Balance", "Month",
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "Payday Loan",
    "Personal Loan", "Student Loan", "Not Specified", "num_loan_types",
]


st.title("Credit Score Predictor")
st.caption("Predicts whether a loan applicant should be approved, rejected, or reviewed.")
st.write(f"pipeline = {pipeline_path}")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=None, placeholder="Age")
    annual_income = st.number_input(
        "Annual Income ($)", min_value=0, max_value=10_000_000, value=None, step=1000,
        placeholder="Annual income",
    )
    monthly_inhand_salary = st.number_input(
        "Monthly Inhand Salary ($)", min_value=0.0, max_value=1_000_000.0, value=None,
        placeholder="Monthly salary",
    )
    num_bank_accounts = st.number_input(
        "Num Bank Accounts", min_value=0, max_value=11, value=None, placeholder="Bank accounts",
    )
    num_credit_card = st.number_input(
        "Num Credit Cards", min_value=0, max_value=11, value=None, placeholder="Credit cards",
    )
    interest_rate = st.number_input(
        "Interest Rate (%)", min_value=0, max_value=34, value=None, placeholder="Interest rate",
    )
    num_of_loan = st.number_input(
        "Num of Loans", min_value=0, max_value=9, value=None, placeholder="Num of loans",
    )

with col2:
    delay_from_due_date = st.number_input(
        "Delay from Due Date (days)", min_value=0, max_value=100, value=None,
        placeholder="Delay days",
    )
    num_of_delayed_payment = st.number_input(
        "Num of Delayed Payments", min_value=0, max_value=28, value=None,
        placeholder="Delayed payments",
    )
    changed_credit_limit = st.number_input(
        "Changed Credit Limit (%)", min_value=-10.0, max_value=30.0, value=None,
        placeholder="Credit limit change",
    )
    num_credit_inquiries = st.number_input(
        "Num Credit Inquiries", min_value=0, max_value=17, value=None,
        placeholder="Credit inquiries",
    )
    outstanding_debt = st.number_input(
        "Outstanding Debt ($)", min_value=0.0, max_value=100_000.0, value=None,
        placeholder="Outstanding debt",
    )
    credit_utilization_ratio = st.number_input(
        "Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=None,
        placeholder="Utilization ratio",
    )
    credit_history_age_months = st.number_input(
        "Credit History Age (months)", min_value=0, max_value=600, value=None,
        placeholder="History age",
    )

with col3:
    total_emi_per_month = st.number_input(
        "Total EMI per Month ($)", min_value=0.0, max_value=3_000.0, value=None,
        placeholder="EMI per month",
    )
    amount_invested_monthly = st.number_input(
        "Amount Invested Monthly ($)", min_value=0.0, max_value=10_000.0, value=None,
        placeholder="Invested monthly",
    )
    monthly_balance = st.number_input(
        "Monthly Balance ($)", min_value=0.0, max_value=10_000.0, value=None,
        placeholder="Monthly balance",
    )
    occupation = st.selectbox(
        "Occupation",
        [
            "Lawyer", "Mechanic", "Teacher", "Developer", "Journalist", "Scientist",
            "Accountant", "Media_Manager", "Architect", "Engineer", "Entrepreneur",
            "Doctor", "Manager", "Musician", "Writer",
        ],
        index=None,
        placeholder="Select occupation",
    )
    credit_mix = st.selectbox(
        "Credit Mix", ["Good", "Standard", "Bad"], index=None, placeholder="Select credit mix",
    )
    payment_of_min_amount = st.selectbox(
        "Payment of Min Amount", ["Yes", "No"], index=None, placeholder="Select payment",
    )
    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        [
            "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
            "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
        ],
        index=None,
        placeholder="Select behaviour",
    )
    month = st.selectbox(
        "Month",
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        index=None,
        placeholder="Select month",
    )

st.subheader("Loan Types")
loan_cols = st.columns(4)
loan_options = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "Payday Loan",
    "Personal Loan", "Student Loan", "Not Specified",
]
loan_flags = {}
for i, loan in enumerate(loan_options):
    with loan_cols[i % 4]:
        loan_flags[loan] = 1 if st.checkbox(loan, value=False) else 0
num_loan_types = sum(loan_flags.values())


if st.button("Predict Credit Score", type="primary"):
    missing = [
        label for label, val in {
            "Age": age,
            "Annual Income": annual_income,
            "Monthly Inhand Salary": monthly_inhand_salary,
            "Num Bank Accounts": num_bank_accounts,
            "Num Credit Cards": num_credit_card,
            "Interest Rate": interest_rate,
            "Num of Loans": num_of_loan,
            "Delay from Due Date": delay_from_due_date,
            "Num of Delayed Payments": num_of_delayed_payment,
            "Changed Credit Limit": changed_credit_limit,
            "Num Credit Inquiries": num_credit_inquiries,
            "Outstanding Debt": outstanding_debt,
            "Credit Utilization Ratio": credit_utilization_ratio,
            "Credit History Age": credit_history_age_months,
            "Total EMI per Month": total_emi_per_month,
            "Amount Invested Monthly": amount_invested_monthly,
            "Monthly Balance": monthly_balance,
            "Occupation": occupation,
            "Credit Mix": credit_mix,
            "Payment of Min Amount": payment_of_min_amount,
            "Payment Behaviour": payment_behaviour,
            "Month": month,
        }.items() if val is None
    ]
    if missing:
        st.warning(f"Isi field berikut = {', '.join(missing)}")
    else:
        features = {
            "Age": float(age), "Occupation": occupation, "Annual_Income": float(annual_income),
            "Monthly_Inhand_Salary": float(monthly_inhand_salary),
            "Num_Bank_Accounts": float(num_bank_accounts),
            "Num_Credit_Card": float(num_credit_card),
            "Interest_Rate": float(interest_rate), "Num_of_Loan": float(num_of_loan),
            "Delay_from_due_date": float(delay_from_due_date),
            "Num_of_Delayed_Payment": float(num_of_delayed_payment),
            "Changed_Credit_Limit": float(changed_credit_limit),
            "Num_Credit_Inquiries": float(num_credit_inquiries),
            "Outstanding_Debt": float(outstanding_debt),
            "Credit_Utilization_Ratio": float(credit_utilization_ratio),
            "Credit_History_Age": float(credit_history_age_months),
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Total_EMI_per_month": float(total_emi_per_month),
            "Amount_invested_monthly": float(amount_invested_monthly),
            "Payment_Behaviour": payment_behaviour,
            "Monthly_Balance": float(monthly_balance),
            "Month": month,
            **loan_flags,
            "num_loan_types": float(num_loan_types),
        }
        input_df = pd.DataFrame([features], columns=feature_names)

        probs = pipeline.predict_proba(input_df)[0]
        pred = int(pipeline.predict(input_df)[0])
        label = label_map[pred]

        if label == "poor":
            st.error(f"### Predicted credit score: **{label}**")
        elif label == "good":
            st.success(f"### Predicted credit score: **{label}**")
        else:
            st.warning(f"### Predicted credit score: **{label}**")

        st.write("Class probabilities:")
        prob_df = pd.DataFrame({
            "class": ["good", "poor", "standard"],
            "probability": probs,
        })
        st.bar_chart(prob_df.set_index("class"))