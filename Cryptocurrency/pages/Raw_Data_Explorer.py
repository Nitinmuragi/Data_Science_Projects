import streamlit as st
import pandas as pd
import os

st.title("Raw Data Explorer")

raw_folder = "data/raw_data"

if not os.path.isdir(raw_folder):
    st.error(f"`{raw_folder}` folder not found.")
else:
    files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]
    if not files:
        st.warning("No CSV files found in `data/raw_data`.")
    else:
        file_choice = st.selectbox("Select raw data file", files)
        path = os.path.join(raw_folder, file_choice)
        df = pd.read_csv(path)
        st.write("Shape:", df.shape)
        st.dataframe(df.head(50))
        st.write("Column summary:")
        st.write(df.describe(include="all"))
