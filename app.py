
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="AI Stock Forecasting",
    page_icon="📈",
    layout="wide"
)

model = tf.keras.models.load_model(
    "lstm_stock_model.h5"
)

scaler = joblib.load(
    "scaler.pkl"
)

st.title(
    "📈 AI Stock Price Forecasting System"
)

st.write(
    "Enter Previous 30 Days Prices"
)

prices = []

for i in range(30):
    value = st.number_input(
        f"Day {i+1}",
        value=100.0
    )
    prices.append(value)

if st.button(
    "Predict Next Day Price"
):

    arr = np.array(
        prices
    ).reshape(-1,1)

    arr = scaler.transform(
        arr
    )

    arr = arr.reshape(
        1,
        30,
        1
    )

    pred = model.predict(
        arr,
        verbose=0
    )

    pred_price = scaler.inverse_transform(
        pred
    )

    st.success(
        f"Predicted Next Day Price: {pred_price[0][0]:.2f}"
    )
