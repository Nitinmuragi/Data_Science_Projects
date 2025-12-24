import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

st.title("Sentiment Analysis")
price_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(price_path):
    st.error("Processed BTC data not found. Please run preprocessing first.")
    st.stop()

df_price = pd.read_csv(price_path, index_col=0, parse_dates=True)

st.subheader("Upload Text/CSV for Sentiment Analysis")

uploaded_file = st.file_uploader(
    "Upload a .txt or .csv file containing text & date",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        raw_text = uploaded_file.read().decode("utf-8").splitlines()
        df = pd.DataFrame({"text": raw_text})
        df["date"] = pd.Timestamp.today()
    else:
        df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    cols = list(df.columns)

    st.markdown("#### Select columns")
    text_col = None
    date_col = None

    possible_text_cols = [c for c in cols if c.lower() in ["text", "tweet", "content", "body"]]
    if possible_text_cols:
        text_col = st.selectbox(
            "Column containing the text",
            cols,
            index=cols.index(possible_text_cols[0])
        )
    else:
        text_col = st.selectbox("Column containing the text", cols)

    possible_date_cols = [c for c in cols if "date" in c.lower() or "time" in c.lower() or "created" in c.lower()]
    date_options = ["<no date column>"] + cols
    if possible_date_cols:
        default_idx = date_options.index(possible_date_cols[0]) if possible_date_cols[0] in date_options else 0
    else:
        default_idx = 0
    date_choice = st.selectbox("Column containing the date (optional)", date_options, index=default_idx)

    if date_choice != "<no date column>":
        date_col = date_choice

    if text_col is None:
        st.error("Please select a text column.")
        st.stop()

    df_std = pd.DataFrame()
    df_std["text"] = df[text_col].astype(str)

    if date_col is not None:
        df_std["date"] = pd.to_datetime(df[date_col])
    else:
        df_std["date"] = pd.Timestamp.today()

    st.write("### Standardised Data")
    st.dataframe(df_std.head())

    st.subheader("Running VADER Sentiment Analysis")

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        score = analyzer.polarity_scores(str(text))
        return score["compound"]

    df_std["sentiment"] = df_std["text"].apply(get_sentiment)

    st.write("### Sentiment Output")
    st.dataframe(df_std[["date", "text", "sentiment"]].head())

    st.subheader("Sentiment Aggregated by Date")

    df_daily = df_std.groupby(df_std["date"].dt.date)["sentiment"].mean().reset_index()
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily.set_index("date", inplace=True)

    df_price.index = pd.to_datetime(df_price.index).tz_localize(None)
    df_daily.index = df_daily.index.tz_localize(None)
    sentiment_aligned = df_price.join(df_daily, how="left")
    sentiment_aligned["sentiment"] = sentiment_aligned["sentiment"].fillna(method="ffill")

    st.subheader("Sentiment vs Bitcoin Price")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sentiment_aligned.index,
        y=sentiment_aligned["Close"],
        name="BTC Price",
        yaxis="y1",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=sentiment_aligned.index,
        y=sentiment_aligned["sentiment"],
        name="Sentiment",
        yaxis="y2",
        mode="lines"
    ))

    fig.update_layout(
        title="Sentiment vs Bitcoin Price",
        yaxis=dict(title="BTC Price", side="left"),
        yaxis2=dict(title="Sentiment Score", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Download Results ----------------
    st.subheader("Download Sentiment Results")

    csv = sentiment_aligned.to_csv().encode("utf-8")
    st.download_button(
        "Download Sentiment Data as CSV",
        csv,
        "sentiment_price_combined.csv",
        "text/csv"
    )

else:
    st.info("Upload a file to run sentiment analysis.")
