import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.title("Volatility Analysis")

processed_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(processed_path):
    st.warning("Processed file not found.")
else:
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    if "Close" not in df.columns:
        st.error("Processed data must contain 'Close'.")
    else:
        window = st.sidebar.number_input("Rolling window (days)", 2, 60, 7)
        returns = df["Close"].pct_change()
        vol = returns.rolling(window).std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vol.index, y=vol, name="Rolling volatility"))
        st.plotly_chart(fig, use_container_width=True)
