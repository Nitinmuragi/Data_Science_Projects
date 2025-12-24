import sys
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models.arima_model import train_arima_model, forecast_arima
from models.sarima_model import train_sarima_model, forecast_sarima
from models.lstm_model import train_lstm_model
from models.prophet_model import forecast_prophet

st.title("Cryptocurrency Price Forecasting")

data = pd.read_csv('data/processed_data/cleaned_btc_data.csv', index_col='Date', parse_dates=True)

st.subheader('Historical Data')
st.line_chart(data['Close'])
st.subheader('ARIMA Forecast')
arima_model = train_arima_model(data['Close'])
arima_forecast = forecast_arima(arima_model)
st.plotly_chart(go.Figure(data=[
    go.Scatter(x=arima_forecast.index, y=arima_forecast, mode='lines', name='ARIMA Forecast'),
    go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data')
]))

st.subheader('SARIMA Forecast')
sarima_model = train_sarima_model(data['Close'])
sarima_forecast = forecast_sarima(sarima_model)
st.plotly_chart(go.Figure(data=[
    go.Scatter(x=sarima_forecast.index, y=sarima_forecast, mode='lines', name='SARIMA Forecast'),
    go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data')
]))

st.subheader('LSTM Forecast')
st.subheader('Prophet Forecast')
prophet_forecast = forecast_prophet(data)

st.plotly_chart(go.Figure(data=[
    go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Prophet Forecast'),
    go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data')
]))
