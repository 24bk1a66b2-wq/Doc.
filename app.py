import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('rf_model.pkl')

# App UI
st.title("Credit Card Fraud Detection")
st.write("Upload transaction data to detect fraud")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if data.shape[1] >= 4:
        st.write("Sample data:")
        st.write(data.head())

        # Predict
        prediction = model.predict(data)
        data['Prediction'] = prediction
        st.write("Prediction results:")
        st.write(data)
        st.success(f"Fraudulent transactions detected: {sum(prediction)}")
    else:
        st.error("CSV must contain at least 4 features")