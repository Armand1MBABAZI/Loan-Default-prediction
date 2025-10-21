# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 20:36:08 2025
@author: user
"""

import pandas as pd
import pickle
import streamlit as st


with open('C:/Users/user/Desktop/Again/Loan prediction.pkl', 'rb') as file:
    model = pickle.load(file)


# Mapping dictionaries
EmploymentType_mapping = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
MaritalStatus_mapping = {"Divorced": 0, "Married": 1, "Single": 2}
LoanPurpose_mapping = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}

# Input fields
st.subheader("Applicant Information")

Age = st.number_input("Age", min_value=18, max_value=100, value=56)
Income = st.number_input("Income", min_value=0, max_value=1000000, value=85994)
LoanAmount = st.number_input("Loan Amount", min_value=0, max_value=1000000, value=50587)
CreditScore = st.number_input("Credit Score", min_value=0, max_value=850, value=520)
MonthsEmployed = st.number_input("Months Employed", min_value=0, max_value=600, value=80)
LoanTerm = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=36)
EmploymentType = st.selectbox("Employment Type", list(EmploymentType_mapping.keys()))
MaritalStatus = st.selectbox("Marital Status", list(MaritalStatus_mapping.keys()))
LoanPurpose = st.selectbox("Loan Purpose", list(LoanPurpose_mapping.keys()))

# Prepare input data for prediction
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

# Prediction button
if st.button("Predict Loan Default Risk"):
    try:
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("High Risk of Default")
            st.write("This application shows a high risk of loan default.")
        else:
            st.success("Low Risk of Default")
            st.write("This application shows a low risk of loan default.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check if the model is compatible with the input features.")
