import streamlit as st
import requests
import subprocess

st.title("Finance Complaint Classifier")

st.header("Identify the financial product of your complaints :moneybag:")

complaint = st.text_area(label="Enter complaint here",
            value="I was trying to apply for a loan to buy a house, but my application keeps getting denied.",
            placeholder="Type your financial compalint...")

data = {"complaint": complaint}

if st.button("Classify Complaint"):
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    prediction = response.text
    st.success(f"Your complaint is referring to a {prediction}.")