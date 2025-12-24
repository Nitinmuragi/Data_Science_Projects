import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_lstm_model(train_series, sequence_length=50, epochs=3):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))

    X, y = create_sequences(scaled, sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(16),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    return model, scaler, scaled, sequence_length

def forecast_lstm(model, scaler, scaled_train, sequence_length, steps):
    last_seq = scaled_train[-sequence_length:].reshape(1, sequence_length, 1)
    preds_scaled = []

    for _ in range(steps):
        pred = model.predict(last_seq, verbose=0)[0][0]
        preds_scaled.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
    return preds.flatten()
