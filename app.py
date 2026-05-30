import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Stock Price Forecasting System",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("📈 AI Stock Price Forecasting System")
st.write("Predict the next day's stock price using the previous 30 days of prices.")

# =====================================================
# CHECK FILES
# =====================================================

required_files = [
    "lstm_stock_model.h5",
    "scaler.pkl"
]

missing_files = [
    f for f in required_files
    if not os.path.exists(f)
]

if missing_files:
    st.error(
        f"Missing files: {', '.join(missing_files)}"
    )
    st.stop()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "lstm_stock_model.h5",
        compile=False
    )

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

try:
    model = load_model()
    scaler = load_scaler()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =====================================================
# USER INPUT
# =====================================================

st.subheader("Enter Previous 30 Days Stock Prices")

prices = []

for i in range(30):
    value = st.number_input(
        f"Day {i+1}",
        min_value=0.0,
        value=100.0,
        step=1.0
    )
    prices.append(value)

# =====================================================
# PREDICT
# =====================================================

if st.button("Predict Next Day Price"):

    try:

        prices_array = np.array(
            prices
        ).reshape(-1, 1)

        prices_scaled = scaler.transform(
            prices_array
        )

        prices_scaled = prices_scaled.reshape(
            1,
            30,
            1
        )

        prediction = model.predict(
            prices_scaled,
            verbose=0
        )

        predicted_price = scaler.inverse_transform(
            prediction
        )[0][0]

        st.success(
            f"📊 Predicted Next Day Price: ₹{predicted_price:.2f}"
        )

    except Exception as e:
        st.error(
            f"Prediction Error: {e}"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption(
    "Deep Learning Assignment 4 - SimpleRNN vs LSTM Stock Forecasting"
)
