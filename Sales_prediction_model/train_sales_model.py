import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv("sales_data.csv")

X = data[['advertising_spend', 'discount']]
y = data['units_sold']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("sales_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Sales Prediction Model Trained Successfully")
