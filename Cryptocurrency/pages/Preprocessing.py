import streamlit as st
import pandas as pd
import os

st.title("Preprocessing Overview")

raw_path = "data/raw_data/BTC-USD.csv"  
processed_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(raw_path):
    st.error(f"Raw file not found: {raw_path}")
    st.stop()

df = pd.read_csv(raw_path)

st.subheader("Raw Data")
st.dataframe(df.head())

df.columns = df.columns.str.lower()

required = ["snapped_at", "price", "market_cap", "total_volume"]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["snapped_at"] = pd.to_datetime(df["snapped_at"], errors="coerce")

if df["snapped_at"].dt.tz is not None:
    df["snapped_at"] = df["snapped_at"].dt.tz_convert(None)

df = df.dropna(subset=["snapped_at"])
df = df.sort_values("snapped_at")

df.rename(columns={"price": "Close"}, inplace=True)

df = df.set_index("snapped_at")

df["7_day_MA"] = df["Close"].rolling(7).mean()

os.makedirs("data/processed_data", exist_ok=True)
df.to_csv(processed_path)

st.success("Processed data saved successfully!")
st.write("Saved to:", processed_path)

st.subheader("Processed Data Preview")
st.dataframe(df.head())
