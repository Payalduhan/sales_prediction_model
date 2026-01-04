 📊 AI-Based Sales Prediction System

An end-to-end machine learning project that predicts future product sales using historical data and business inputs such as advertising spend and discount rates. The system visualizes past sales trends and forecasts future sales through an interactive Streamlit dashboard.

 🚀 Features
- 📈 Visualizes historical sales trends
- 🔮 Predicts future sales using Machine Learning
- 🎯 Takes business inputs (Advertising Spend & Discount)
- 📊 Displays past vs predicted future sales on the same graph
- 🖥️ Interactive and user-friendly Streamlit interface

 🧠 Technologies Used
- **Python**
- **Pandas & NumPy** – Data processing
- **Matplotlib** – Data visualization
- **Scikit-learn** – Machine Learning model
- **Streamlit** – Web application
- **Pickle** – Model serialization

📁 Project Structure
AI_Sales_Prediction_System/
│
├── app.py # Streamlit application
├── train_sales_model.py # Model training script
├── sales_model.pkl # Trained ML model
├── sales_data.csv # Historical sales dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation

▶️ How to Run the Project

Step 1: Install dependencies
pip install -r requirements.txt
Step 2: Train the model
python train_sales_model.py
Step 3: Run Streamlit app
streamlit run app.py

🧪 Future Enhancements
Add time-series models (ARIMA / LSTM)

Predict sales for each future month dynamically

Include feature importance visualization

Deploy on Streamlit Cloud

👤 Author
Payal Duhan

⭐ If you like this project, feel free to star the repository!







