import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model
with open("sales_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
data = pd.read_csv("sales_data.csv")
data['date'] = pd.to_datetime(data['date'])

st.set_page_config(page_title="AI Sales Prediction System", layout="centered")

st.title("📊 AI-Based Sales Prediction System")
st.markdown("Predict future sales using Machine Learning and visualize trends")

# ------------------- Past Sales Graph -------------------
st.subheader("📈 Past Sales Trend")

fig, ax = plt.subplots()
ax.plot(data['date'], data['units_sold'], marker='o')
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.set_title("Past Sales Data")
st.pyplot(fig)

st.subheader("🔮 Future Sales Input")

future_ad_spend = st.slider(
    "Future Advertising Spend",
    min_value=500,
    max_value=1200,
    value=800
)

future_discount = st.slider(
    "Future Discount (%)",
    min_value=0,
    max_value=50,
    value=20
)

# ------------------- User Input -------------------
st.subheader("🧮 Predict Future Sales")

ad_spend = st.number_input("Advertising Spend (₹)", min_value=0)
discount = st.slider("Discount (%)", 0, 50)

if st.button("Predict Sales"):
    input_data = np.array([[ad_spend, discount]])
    prediction = model.predict(input_data)[0]

    st.success(f"📦 Predicted Units Sold: **{int(prediction)}**")
st.info("Past sales data is loaded from historical company records. User inputs are used to predict future sales.")

# ------------------- Future Sales Visualization -------------------
st.subheader("🔮 Future Sales Prediction Visualization")

future_sales = model.predict(data[['advertising_spend', 'discount']])

fig2, ax2 = plt.subplots()
ax2.plot(data['date'], data['units_sold'], label="Actual Sales")
ax2.plot(data['date'], future_sales, linestyle='dashed', label="Predicted Sales")
ax2.set_title("Actual vs Predicted Sales")
ax2.legend()
st.pyplot(fig2)

# Number of months to predict
future_months = 6

# Create future dates
last_date = data['date'].max()
future_dates = pd.date_range(
    start=last_date,
    periods=future_months + 1,
    freq='M'
)[1:]

# Load trained model
model = pickle.load(open("sales_model.pkl", "rb"))

# Prepare future input
future_X = np.array([
    [future_ad_spend, future_discount]
] * future_months)

# Predict future sales
future_sales = model.predict(future_X)

# Plot past + future sales
fig, ax = plt.subplots()

ax.plot(
    data['date'],
    data['units_sold'],
    marker='o',
    label="Past Sales"
)

ax.plot(
    future_dates,
    future_sales,
    linestyle='--',
    marker='o',
    label="Predicted Future Sales"
)

ax.set_title("Past vs Predicted Future Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
ax.grid(True)

st.pyplot(fig)
