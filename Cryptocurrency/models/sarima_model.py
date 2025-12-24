from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima_model(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

def forecast_sarima(model_fit, steps=30):
    fc = model_fit.get_forecast(steps=steps)
    return fc.predicted_mean
