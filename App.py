# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 20:36:08 2025
@author: user
"""

import pandas as pd
import pickle
import streamlit as st
import os

st.title("💰 Loan Default Prediction App")

# Enhanced model loading with multiple fallback options
def load_model():
    # Try multiple possible file names and paths
    possible_files = [
        'Loan prediction.pkl',
        'Loan prediction model1.pkl',
        'model.pkl',
        './Loan prediction.pkl',
        './Loan prediction model1.pkl',
        'models/Loan prediction.pkl',
        'models/Loan prediction model1.pkl'
    ]
    
    for file_path in possible_files:
        try:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            st.success(f"✅ Model loaded successfully from: {file_path}")
            return model
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ Found {file_path} but error loading: {str(e)}")
            continue
    
    # If no file found, show error and available files
    st.error("❌ Model file not found. Please check:")
    st.write("1. The model file exists in your project")
    st.write("2. The filename matches exactly")
    st.write("3. The file is uploaded to Streamlit Cloud")
    
    # Show available files for debugging
    st.info("📁 Files in your project directory:")
    try:
        files = os.listdir('.')
        if files:
            for file in files:
                st.write(f"- {file}")
        else:
            st.write("No files found in directory")
    except Exception as e:
        st.write(f"Unable to list files: {str(e)}")
    
    return None

# Load the model
model = load_model()

# Only show the input form if model is loaded successfully
if model is not None:
    # Mapping dictionaries
    EmploymentType_mapping = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
    MaritalStatus_mapping = {"Divorced": 0, "Married": 1, "Single": 2}
    LoanPurpose_mapping = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}

    # Input fields
    st.subheader("Applicant Information")

    Age = st.number_input("Age", min_value=18, max_value=100, value=56)
    Income = st.number_input("Income", min_value=0, max_value=1000000, value=85994)
    LoanAmount = st.number_input("Loan Amount", min_value=0, max_value=1000000, value=50587)
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=520)
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
                st.error("🚨 High Risk of Default")
                st.write("This application shows a high risk of loan default.")
            else:
                st.success("✅ Low Risk of Default")
                st.write("This application shows a low risk of loan default.")
                
            # Try to show probability scores if available
            try:
                probability = model.predict_proba(input_data)
                st.subheader("Probability Scores")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probability of No Default", f"{probability[0][0]:.2%}")
                with col2:
                    st.metric("Probability of Default", f"{probability[0][1]:.2%}")
            except:
                st.info("Probability scores not available for this model")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check if the model is compatible with the input features.")
else:
    st.warning("🔧 Please upload your model file to use the prediction app.")

# Add instructions for fixing the issue
with st.expander("🔧 How to fix this issue"):
    st.markdown("""
    **To fix the model file not found error:**

    1. **Upload the model file to Streamlit Cloud:**
       - Go to your app on Streamlit Cloud
       - Click 'Manage app' → 'Settings' → 'Advanced'
       - Upload your `.pkl` file

    2. **Ensure correct filename:**
       - Make sure your model file is named exactly as expected
       - Common names: `Loan prediction.pkl` or `Loan prediction model1.pkl`

    3. **Check file location:**
       - The model file should be in the same directory as your script
       - Or in a `models/` subdirectory

    4. **Verify file format:**
       - Ensure the `.pkl` file is not corrupted
       - Try loading it locally first to verify

    **Required project structure:**
    ```
    your-project/
    ├── app.py          (this script)
    ├── Loan prediction.pkl  (your model file)
    └── requirements.txt
    ```
    """)
