import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from models.sarima_model import train_sarima_model, forecast_sarima

st.title("SARIMA Forecast")

processed_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(processed_path):
    st.warning("Processed file not found.")
else:
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    if "Close" not in df.columns:
        st.error("Processed data must contain a 'Close' column.")
    else:
        st.sidebar.header("SARIMA parameters")
        p = st.sidebar.number_input("p", 0, 5, 1)
        d = st.sidebar.number_input("d", 0, 2, 1)
        q = st.sidebar.number_input("q", 0, 5, 1)
        P = st.sidebar.number_input("P", 0, 5, 1)
        D = st.sidebar.number_input("D", 0, 2, 1)
        Q = st.sidebar.number_input("Q", 0, 5, 1)
        s = st.sidebar.number_input("Seasonal period (s)", 1, 365, 7)
        steps = st.sidebar.number_input("Forecast steps", 1, 365, 30)

        series = df["Close"]

        if st.button("Run SARIMA"):
            with st.spinner("Fitting SARIMA model..."):
                model_fit = train_sarima_model(
                    series,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                )
                forecast = forecast_sarima(model_fit, steps=steps)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical"))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)
