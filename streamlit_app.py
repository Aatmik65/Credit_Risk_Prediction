import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Prediction App")
@st.cache_data
def load_data():
    return pd.read_csv("credit_data_synthetic.csv")
df = load_data()
import numpy as np
if 'EmploymentYears' not in df.columns:
    np.random.seed(42)
    df['EmploymentYears'] = np.random.randint(0, 20, size=len(df))
X = df[['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentYears']]
y = df['Default']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
@st.cache_resource
def train_model():
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model

model = train_model()
st.sidebar.header("📋 Enter Customer Details")
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Monthly Income (INR)", 10000, 1000000, step=1000)
loan = st.sidebar.number_input("Loan Amount (INR)", 1000, 1000000, step=1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
employment_years = st.sidebar.slider("Employment Years", 0, 40, 2)

if st.sidebar.button("🔍 Predict Credit Risk"):
    input_data = pd.DataFrame([[age, income, loan, credit_score, employment_years]],
                              columns=['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentYears'])
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    score = (1 - prob) * 100

    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of Default:** {prob:.2f}")
    st.write(f"**Credit Risk Score:** {score:.1f} / 100")
    if score > 70:
        st.success("🟢 Low Risk: Likely to Repay")
    else:
        st.error("🔴 High Risk: Likely to Default")
