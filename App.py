# -*- coding: utf-8 -*-
"""
Loan Default Prediction App for Streamlit Cloud
Created on Sat Oct 18 20:36:08 2025
@author: user
"""

import pandas as pd
import streamlit as st
import joblib
import pickle
import os

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Loan Default Prediction", page_icon="💰")
st.title("💰 Loan Default Risk Prediction App")
st.write("Predict whether a loan applicant is likely to default using a trained ML model.")

# -----------------------------
# Model loading
# -----------------------------
model = None
model_path = "Loan prediction.pkl"  # Must be in the same GitHub folder as this script

if os.path.exists(model_path):
    try:
        # Try loading with joblib first
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully (joblib).")
    except Exception:
        try:
            # Fallback to pickle
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            st.success("✅ Model loaded successfully (pickle).")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()
else:
    st.error(f"❌ Model file not found at: {model_path}")
    st.info("Please upload 'Loan prediction.pkl' to your GitHub repository.")
    st.stop()

# -----------------------------
# Input mappings
# -----------------------------
EmploymentType_mapping = {
    "Full-time": 0,
    "Part-time": 1,
    "Self-employed": 2,
    "Unemployed": 3
}
MaritalStatus_mapping = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}
LoanPurpose_mapping = {
    "Auto": 0,
    "Business": 1,
    "Education": 2,
    "Home": 3,
    "Other": 4
}

# -----------------------------
# Input form
# -----------------------------
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    Income = st.number_input("Monthly Income ($)", min_value=0, max_value=1_000_000, value=50000)
    LoanAmount = st.number_input("Loan Amount ($)", min_value=0, max_value=1_000_000, value=20000)
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

with col2:
    MonthsEmployed = st.number_input("Months Employed", min_value=0, max_value=600, value=36)
    LoanTerm = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=60)
    EmploymentType = st.selectbox("Employment Type", list(EmploymentType_mapping.keys()))
    MaritalStatus = st.selectbox("Marital Status", list(MaritalStatus_mapping.keys()))
    LoanPurpose = st.selectbox("Loan Purpose", list(LoanPurpose_mapping.keys()))

# -----------------------------
# Prepare data
# -----------------------------
input_data = [[
    Age,
    Income,
    LoanAmount,
    CreditScore,
    MonthsEmployed,
    LoanTerm,
    EmploymentType_mapping[EmploymentType],
    MaritalStatus_mapping[MaritalStatus],
    LoanPurpose_mapping[LoanPurpose]
]]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Loan Default Risk"):
    try:
        prediction = model.predict(input_data)
        result = prediction[0]

        if result == 1:
            st.error("⚠️ High Risk of Loan Default")
            st.write("This applicant shows a **high risk** of defaulting on the loan.")
        else:
            st.success("✅ Low Risk of Loan Default")
            st.write("This applicant shows a **low risk** of loan default.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Ensure the model was trained with the same input feature order.")
