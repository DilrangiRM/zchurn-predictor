import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model_data = joblib.load('model/churn_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

# UI Title
st.title("📱 Gen Z Telecom Churn Predictor")
st.markdown("Enter customer information to predict churn.")

# Input fields
user_input = []
for col in features:
    value = st.text_input(f"{col}", "")
    user_input.append(value)

# Predict button
if st.button("Predict"):
    try:
        # Create dataframe from input
        input_df = pd.DataFrame([user_input], columns=features)
        input_df = input_df.apply(pd.to_numeric)

        # Scale if needed
        if scaler:
            input_df = scaler.transform(input_df)

        # Predict churn
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'⚠️ Will Churn' if prediction == 1 else '✅ Will Not Churn'}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")