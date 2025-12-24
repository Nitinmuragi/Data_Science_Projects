import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.arima_model import train_arima_model, forecast_arima
from models.sarima_model import train_sarima_model, forecast_sarima
from models.prophet_model import train_prophet_model, forecast_prophet
from models.lstm_model import train_lstm_model, forecast_lstm

st.title("Model Comparison Dashboard")

processed_path = "data/processed_data/cleaned_btc_data.csv"

if not st.session_state.get("df_loaded", False):
    if not processed_path or not os.path.exists(processed_path):
        st.error("Processed data not found. Please preprocess the data first.")
    else:
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        st.session_state["df"] = df
        st.session_state["df_loaded"] = True

df = st.session_state.get("df")

if df is None:
    st.stop()

if "Close" not in df.columns:
    st.error("Processed data must contain a 'Close' column.")
    st.stop()

st.sidebar.header("Model Comparison Settings")
train_size = st.sidebar.slider("Training Percentage", 50, 95, 80) / 100

split_index = int(len(df) * train_size)
train = df["Close"].iloc[:split_index]
test = df["Close"].iloc[split_index:]

steps = len(test)

st.write(f"Training samples: {len(train)}, Testing samples: {len(test)}")


def compute_errors(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    mape = np.mean(np.abs((true - predicted) / true)) * 100
    return rmse, mae, mape



results = {}

with st.spinner("Running ARIMA..."):
    try:
        arima_model = train_arima_model(train)
        arima_pred = forecast_arima(arima_model, steps=steps)
        arima_pred.index = test.index  # align index
        results["ARIMA"] = compute_errors(test, arima_pred)
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")

with st.spinner("Running SARIMA..."):
    try:
        sarima_model = train_sarima_model(train)
        sarima_pred = forecast_sarima(sarima_model, steps=steps)
        sarima_pred.index = test.index
        results["SARIMA"] = compute_errors(test, sarima_pred)
    except Exception as e:
        st.warning(f"SARIMA failed: {e}")


with st.spinner("Running Prophet..."):
    try:
        model = train_prophet_model(train)
        full_forecast = forecast_prophet(model, steps=steps)
        prophet_pred = full_forecast["yhat"].iloc[-steps:].values
        
        results["Prophet"] = compute_errors(test, prophet_pred)

    except Exception as e:
        st.warning(f"Prophet failed: {e}")

with st.spinner("Running LSTM..."):
    try:
        lstm_model, scaler, scaled_train, seq_len = train_lstm_model(train)
        lstm_pred = forecast_lstm(lstm_model, scaler, scaled_train, seq_len, steps)

        results["LSTM"] = compute_errors(test, lstm_pred)

    except Exception as e:
        st.warning(f"LSTM failed: {e}")
        results["LSTM"] = (None, None, None)


st.subheader("Model Error Metrics")

df_results = pd.DataFrame(results, index=["RMSE", "MAE", "MAPE"]).T
st.dataframe(df_results)

st.subheader("Error Comparison Chart")

fig = go.Figure()

for metric in ["RMSE", "MAE", "MAPE"]:
    fig.add_trace(go.Bar(
        x=df_results.index,
        y=df_results[metric],
        name=metric
    ))

fig.update_layout(
    barmode='group',
    xaxis_title="Model",
    yaxis_title="Error",
    title="Model Error Comparison"
)

st.plotly_chart(fig, use_container_width=True)
