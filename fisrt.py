import streamlit as st
import pickle
import numpy as np

# Load model (correct path)
model = pickle.load(open(r'C:\Users\SAHAJ\Desktop\pythan\linear_regression_model.pkl', 'rb'))

# Title
st.title("Salary Prediction App")

# Input from user
years = st.number_input("Enter Years of Experience")

# Prediction button
if st.button("Predict Salary"):
    result = model.predict([[years]])
    st.success(f"Predicted Salary: ₹{result[0]:,.2f}")