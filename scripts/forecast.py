import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def generate_forecast():
    # Load stock data
    df = pd.read_csv('historical_data.csv', index_col='Date', parse_dates=True)
    
    forecasts = {}
    for stock in ['TSLA', 'SPY', 'BND']:
        train_data = df[stock].dropna()
        model = ARIMA(train_data, order=(5,1,0))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=30)  # Predict next 30 days
        forecasts[stock] = forecast.tolist()

    return forecasts
