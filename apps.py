import streamlit as st
import numpy as np
import joblib
st.markdown("[View Code](https://github.com/dnyaneshwari8/mobile-price-prediction)")

model = joblib.load('xgb_mobile_price_model.pkl')
scaler = joblib.load('xgb_scaler.pkl')

st.set_page_config(page_title=" Mobile Price Predictor", layout="centered")
st.title(" Mobile Price Prediction App")
st.markdown("Predict smartphone launch price based on specifications")

weight = st.number_input(" Weight (grams)", min_value=80.0, max_value=600.0, step=1.0)
ram = st.number_input(" RAM (GB)", min_value=1, max_value=24, step=1)
front_cam = st.number_input(" Front Camera (MP)", min_value=1.0, max_value=100.0, step=1.0)
back_cam = st.number_input(" Back Camera (MP)", min_value=1.0, max_value=200.0, step=1.0)
battery = st.number_input(" Battery Capacity (mAh)", min_value=1000, max_value=10000, step=100)
screen = st.number_input(" Screen Size (inches)", min_value=4.0, max_value=12.0, step=0.1)
year = st.selectbox(" Launch Year", list(range(2014, 2026)))
storage = st.selectbox(" Storage (GB)", [16, 32, 64, 128, 256, 512, 1024])

# Encoded processor and company mappings
processor_options = {
    0: 'Apple Bionic',
    1: 'Google Tensor',
    2: 'HiSilicon Kirin',
    3: 'MediaTek Dimensity',
    4: 'MediaTek Helio',
    5: 'Qualcomm Snapdragon',
    6: 'Samsung Exynos',
    7: 'Unisoc'
}
company_options = {
    0: 'Apple', 1: 'Google', 2: 'Honor', 3: 'Huawei', 4: 'Infinix',
    5: 'Iqoo', 6: 'Lenovo', 7: 'Motorola', 8: 'Nokia', 9: 'Oneplus',
    10: 'Oppo', 11: 'Poco', 12: 'Realme', 13: 'Samsung', 14: 'Sony',
    15: 'Tecno', 16: 'Vivo', 17: 'Xiaomi'
}

# User selects
selected_processor = st.selectbox(" Processor", list(processor_options.values()))
selected_company = st.selectbox(" Company", list(company_options.values()))

# Convert selections to encoded values
processor_encoded = [k for k, v in processor_options.items() if v == selected_processor][0]
company_encoded = [k for k, v in company_options.items() if v == selected_company][0]

# Feature array
features = np.array([[weight, ram, front_cam, back_cam, battery, screen,
                      year, storage, processor_encoded, company_encoded]])

# Scale features
scaled_features = scaler.transform(features)

# Prediction
if st.button("Predict Price "):
    log_price = model.predict(scaled_features)
    price = np.expm1(log_price)  # reverse log1p
    corrected_price = price[0] * 0.94  # Optional correction factor
    st.success(f"## Estimated Launch Price: ₹{int(corrected_price):,}")
