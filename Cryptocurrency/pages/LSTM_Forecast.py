import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("LSTM Forecast")

processed_path = "data/processed_data/cleaned_btc_data.csv"

if not os.path.exists(processed_path):
    st.error("Processed data not found.")
    st.stop()

df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

if "Close" not in df.columns:
    st.error("Processed data must contain a 'Close' column.")
    st.stop()

data = df["Close"].values.reshape(-1, 1)

st.sidebar.header("LSTM Settings")

sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 50)
epochs = st.sidebar.slider("Epochs", 1, 10, 3)  
future_days = st.sidebar.slider("Forecast Days", 1, 60, 30)

max_training_rows = st.sidebar.slider("Use last N rows for training", 500, len(data), 1000)

data = data[-max_training_rows:]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(16),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

st.subheader("Training LSTM model...")

progress = st.progress(0)
for epoch in range(epochs):
    model.fit(X, y, epochs=1, batch_size=64, verbose=0)
    progress.progress((epoch + 1) / epochs)

st.success("Model training completed!")

train_pred = model.predict(X)
train_pred = scaler.inverse_transform(train_pred)

actual = data[sequence_length:]

future_input = scaled_data[-sequence_length:]
future_input = future_input.reshape(1, sequence_length, 1)

future_predictions = []

for _ in range(future_days):
    pred = model.predict(future_input)[0][0]
    future_predictions.append(pred)
    future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

st.subheader("LSTM Prediction vs Actual")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-len(actual):], y=actual.flatten(), mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=df.index[-len(actual):], y=train_pred.flatten(), mode="lines", name="LSTM Prediction"))

future_idx = pd.date_range(df.index[-1], periods=future_days+1, freq="D")[1:]
fig.add_trace(go.Scatter(x=future_idx, y=future_predictions.flatten(), mode="lines", name="Future Forecast"))

fig.update_layout(title="LSTM Forecast")

st.plotly_chart(fig, use_container_width=True)
