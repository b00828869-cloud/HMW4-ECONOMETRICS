import streamlit as st

st.set_page_config(
    page_title="HAR Realized Variance App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 HAR Model — Realized Variance Forecasting")

st.markdown("""
This app answers the Financial Econometrics assignment step by step.

Each tab corresponds to one question: data preparation, realized variance, HAR model, forecasting, machine learning and Granger causality.
""")