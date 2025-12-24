import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from models.arima_model import train_arima_model, forecast_arima

st.title("ARIMA Forecast")

processed_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(processed_path):
    st.warning("Processed file not found. Go to 'Preprocessing' page first.")
else:
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    if "Close" not in df.columns:
        st.error("Processed data must contain a 'Close' column.")
    else:
        st.sidebar.header("ARIMA parameters")
        p = st.sidebar.number_input("p", min_value=0, max_value=5, value=1)
        d = st.sidebar.number_input("d", min_value=0, max_value=2, value=1)
        q = st.sidebar.number_input("q", min_value=0, max_value=5, value=1)
        steps = st.sidebar.number_input("Forecast steps", min_value=1, max_value=365, value=30)

        series = df["Close"]

        if st.button("Run ARIMA"):
            with st.spinner("Fitting ARIMA model..."):
                model_fit = train_arima_model(series, order=(p, d, q))
                forecast = forecast_arima(model_fit, steps=steps)

            st.success("Model fitted and forecast generated.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Historical"))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode="lines", name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)
