import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Crypto Time Series Dashboard",
    layout="wide"
)

st.title(" Cryptocurrency Time Series Analysis Dashboard")

st.markdown(
    """
This dashboard explores cryptocurrency time-series using:

- **ARIMA / SARIMA** (classical statistical models)  
- **LSTM** (deep learning)  
- **Prophet** (additive trend + seasonality)  

Use the sidebar to switch between pages:
- Raw data, preprocessing, price/volume plots  
- Model-specific forecast pages  
- Volatility, sentiment, and model comparison
"""
)

processed_path = "data/processed_data/cleaned_btc_data.csv"

if os.path.exists(processed_path):
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    st.subheader("Quick preview of processed data")
    st.dataframe(df.head())
else:
    st.warning(
        "Processed file not found at "
        "`data/processed_data/cleaned_btc_data.csv`. "
        "Please run your preprocessing script or place the cleaned CSV there."
    )
