from prophet import Prophet
import pandas as pd

def train_prophet_model(series):
    df = pd.DataFrame({
        "ds": series.index,
        "y": series.values
    })

    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_convert(None)

    model = Prophet()
    model.fit(df)
    return model

def forecast_prophet(model, steps=30):
    future = model.make_future_dataframe(periods=steps)

    if future['ds'].dt.tz is not None:
        future['ds'] = future['ds'].dt.tz_convert(None)

    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

