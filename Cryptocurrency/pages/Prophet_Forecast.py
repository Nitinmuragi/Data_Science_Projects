import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from models.prophet_model import train_prophet_model, forecast_prophet

st.title("Prophet Forecast")

processed_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(processed_path):
    st.warning("Processed file not found.")
else:
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    if "Close" not in df.columns:
        st.error("Processed data must contain 'Close'.")
    else:
        steps = st.sidebar.number_input("Forecast days", 1, 365, 60)

        if st.button("Run Prophet"):
            series = df["Close"]
            model = train_prophet_model(series)
            forecast = forecast_prophet(model, steps=steps)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical"))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)
