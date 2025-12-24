import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.title("Price and Volume Visualisation")

processed_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(processed_path):
    st.warning("Processed file not found. Go to 'Preprocessing' page first.")
else:
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    st.subheader("Close price")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    st.plotly_chart(fig_price, use_container_width=True)

    if "Volume" in df.columns:
        st.subheader("Volume")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
        st.plotly_chart(fig_vol, use_container_width=True)
