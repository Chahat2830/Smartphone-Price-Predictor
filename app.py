import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
# Matplotlib is imported here for the optional Feature Importance plot
import matplotlib.pyplot as plt

# --- Load trained Random Forest model and scaler ---
try:
    with open("rf_model.pkl", "rb") as file:
        rf_model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model or scaler files (rf_model.pkl, scaler.pkl) not found. Please ensure they are in the same directory as app.py.")
    st.stop()


# --- App Title ---
st.title("📱 Smartphone Price Predictor")
st.write("Predict smartphone price (INR) based on specifications")

# --- Sidebar Inputs ---
st.sidebar.header("Phone Specifications")

# Categorical inputs
brand_options = ['Samsung', 'Xiaomi', 'Apple', 'Realme', 'OnePlus', 'Others']
os_options = ['Android', 'iOS']
age_options = ['New', 'Mid', 'Old']

brand = st.sidebar.selectbox("Brand", brand_options)
os = st.sidebar.selectbox("Operating System", os_options)
phone_age_category = st.sidebar.selectbox("Phone Age Category", age_options)

# Numeric inputs (Adjusted max/min based on data capping in notebook)
weight = st.sidebar.number_input("Weight (g)", min_value=130.0, max_value=235.0, value=180.0, step=1.0)
ppi = st.sidebar.number_input("Pixel Density (PPI)", min_value=167.0, max_value=575.0, value=400.0, step=0.1)
video_capabilities = st.sidebar.number_input("Video Capabilities (Score 0-4)", min_value=0.0, max_value=4.0, value=3.0, step=1.0)
high_fps_support = st.sidebar.number_input("High FPS Support (Score 0-3)", min_value=0.0, max_value=3.0, value=2.0, step=1.0)
performance_score = st.sidebar.number_input("Performance Score", min_value=545.0, max_value=1243.0, value=800.0, step=0.1)
years_since_launch = st.sidebar.number_input("Years Since Launch", min_value=0.0, max_value=8.0, value=1.0, step=1.0)


# --- Prepare input DataFrame ---

# Initialize all features needed for the model in the correct order
input_data = {
    # Numeric features
    'weight(g)': [weight],
    'ppi': [ppi],
    'video_capabilities': [video_capabilities],
    'high_fps_support': [high_fps_support],
    'performance_score': [performance_score],
    'years_since_launch': [years_since_launch],

    # One-hot encoded features (initialize all to 0) - Must match the exact features output by the notebook
    # Note: 'brand_Apple', 'os_Android', 'phone_age_category_New' are the dropped first columns (base categories)
    'brand_OnePlus': [0], 
    'brand_Others': [0], 
    'brand_Realme': [0], 
    'brand_Samsung': [0], 
    'brand_Xiaomi': [0],
    'os_iOS': [0],
    'phone_age_category_Mid': [0], 
    'phone_age_category_Old': [0]
}

# Set selected categories to 1
if brand in input_data and brand not in ['Apple', 'Others']: 
    input_data[f"brand_{brand}"][0] = 1
elif brand == 'Others':
    input_data['brand_Others'][0] = 1
# Apple is the base category (all its dummy features remain 0)

if os == 'iOS':
    input_data['os_iOS'][0] = 1

if phone_age_category == 'Mid':
    input_data['phone_age_category_Mid'][0] = 1
elif phone_age_category == 'Old':
    input_data['phone_age_category_Old'][0] = 1

input_df = pd.DataFrame(input_data)

# Define numeric columns for scaling
numeric_cols = ['weight(g)', 'ppi', 'video_capabilities', 'high_fps_support', 'performance_score', 'years_since_launch']

# Scale numeric features using the loaded scaler
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# --- Predict price ---
if st.button("Predict Price"):
    predicted_price = rf_model.predict(input_df)[0]
    st.success(f"💰 Predicted Price: ₹ {predicted_price:,.0f}")

# --- Optional: Feature Importance ---
if st.checkbox("Show Feature Importance"):
    importances = rf_model.feature_importances_
    features = input_df.columns
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(range(len(features)), importances[indices])
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features[indices], rotation=90)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)
