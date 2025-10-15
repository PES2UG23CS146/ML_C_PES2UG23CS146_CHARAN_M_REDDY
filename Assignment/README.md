# S&P 500 ML Predictor

Machine Learning models to predict **S&P 500 price levels** (regression) and **spread direction** (classification) using 20 years of historical data. This project includes a **Streamlit GUI** for interactive predictions and visualization.

---<br><br><br>

## Features

- Predict **next-day S&P 500 closing price** using Random Forest and Gradient Boosting regressors.
- Predict **spread direction** (Up/Down) using Random Forest and Gradient Boosting classifiers.
- **18 engineered features** including SMA, RSI, MACD, Bollinger Bands, ATR, and more.
- **Interactive Streamlit dashboard** for live predictions and historical trends.

---<br><br><br>

## Installation

1. **Clone the repository**<br>
> git clone https://github.com/PES2UG23CS146/ML_C_PES2UG23CS146_CHARAN_M_REDDY<br><br>
2. **Enter the directory**<br>
> cd Assignment<br><br>
3. **Install Dependencies**<br>
> pip install yfinance==0.2.66 pandas==2.2.2 numpy==2.2.6 scikit-learn streamlit==1.50.0 plotly==6.3.1 joblib==1.4.0<br><br><br>

## Run all Codes in Jupyter NB

> Make sure there are no errors .

## Run

> streamlit run app.py


