import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf

# Define the stock tickers
tickers = ['TSLA', 'BND', 'SPY']

# Download historical data
data = yf.download(tickers, start='2015-01-01', end='2025-01-01')

# Check the columns to understand the data structure
print(data.columns)
# Check basic statistics
print(data.describe())

# Check data types
print(data.dtypes)

# Check for missing values
print(data.isnull().sum())
# Option 1: Interpolate missing values
data = data.interpolate(method='linear')

# Option 2: Fill forward/backward
# data = data.fillna(method='ffill').fillna(method='bfill')

# Double-check for any remaining missing values
print(data.isnull().sum())
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

# Compare original and scaled data
print(scaled_data.head())
# Flatten the columns if multi-indexed
data.columns = ['_'.join(col).strip() for col in data.columns]

print(data.columns)
# Plotting the closing prices
plt.figure(figsize=(10, 6))
for ticker in tickers:
    plt.plot(data.index, data[f'Close_{ticker}'], label=ticker)

plt.title('Closing Prices of TSLA, BND, SPY Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
 #Flatten the multi-index columns if needed
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

# Display the columns to verify the structure
print("Original Columns:", data.columns)

# Rename 'Close' columns to match ticker names directly
rename_mapping = {
    'Close_TSLA': 'TSLA',
    'Close_BND': 'BND',
    'Close_SPY': 'SPY'
}

# Only rename if the expected columns exist
available_columns = set(data.columns)
for original, new_name in rename_mapping.items():
    if original in available_columns:
        data = data.rename(columns={original: new_name})

# Verify column names after renaming
print("Renamed Columns:", data.columns)

# Calculate daily percentage change (returns)
returns = data[['TSLA', 'BND', 'SPY']].pct_change().dropna()

# Verify the columns of the 'returns' DataFrame
print("Returns Columns:", returns.columns)
#Plotting the Daily Percentage Changes:
plt.figure(figsize=(10, 6))
for ticker in tickers:
    plt.plot(returns.index, returns[ticker], label=ticker)

plt.title('Daily Percentage Change of TSLA, BND, SPY')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.legend()
plt.show()

# Volatility Analysis: Rolling Means & Standard Deviations
window_size = 30  # 30-day rolling window

plt.figure(figsize=(10, 6))
for ticker in tickers:
    rolling_std = returns[ticker].rolling(window=window_size).std()
    plt.plot(rolling_std, label=f'{ticker} Rolling Std (30 days)')

plt.title('30-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()
# Plotting boxplots to visualize outliers
plt.figure(figsize=(8, 6))
sns.boxplot(data=returns)
plt.title('Outlier Analysis of Daily Returns')
plt.xlabel('Assets')
plt.ylabel('Daily Returns')
plt.show()

# Analyzing extreme values
for ticker in tickers:
    outliers = returns[(returns[ticker] > returns[ticker].quantile(0.99)) | 
                       (returns[ticker] < returns[ticker].quantile(0.01))]
    print(f'Outliers for {ticker}:')
    print(outliers[ticker].describe())
# Decomposition for Tesla stock
decompose_result = seasonal_decompose(data['TSLA'], model='multiplicative', period=252)

# Plotting the decomposition
plt.figure(figsize=(10, 8))
decompose_result.plot()
plt.show()
 #Key Financial Metrics: VaR & Sharpe Ratio
# Value at Risk (VaR)
# 95% VaR assuming normal distribution
var_95 = returns['TSLA'].quantile(0.05)
print(f"95% Value at Risk (VaR) for TSLA: {var_95:.4f}")
#Sharpe Ratio
# Assuming a risk-free rate of 3% annualized
risk_free_rate = 0.03
trading_days = 252

sharpe_ratio = (returns.mean() * trading_days - risk_free_rate) / (returns.std() * np.sqrt(trading_days))
print(f"Sharpe Ratio for TSLA: {sharpe_ratio['TSLA']:.4f}")
# Define the training and testing split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

print(f'Training data shape: {train_data.shape}')
print(f'Testing data shape: {test_data.shape}')
# Select only the 'TSLA' column for closing prices
train_data = train_data[['TSLA']]

# Check the data shape and preview the data
print(train_data.shape)  # Should be (n_samples, 1)
print(train_data.head())

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pmdarima import auto_arima

# Use the correct 'TSLA' column for forecasting
arima_model = auto_arima(train_data['TSLA'],  # Pass as a 1D array
                         seasonal=False, 
                         trace=True, 
                         suppress_warnings=True,
                         stepwise=True)

print(arima_model.summary())
# Train ARIMA with the best found parameters
model_arima = ARIMA(train_data, order=arima_model.order)
model_arima_fit = model_arima.fit()

# Forecast the test set length
forecast_arima = model_arima_fit.forecast(steps=len(test_data))
test_data['ARIMA_Prediction'] = forecast_arima

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['TSLA'], label='Training Data')
plt.plot(test_data.index, test_data['TSLA'], label='Actual Price')
plt.plot(test_data.index, test_data['ARIMA_Prediction'], label='ARIMA Prediction')
plt.title('Tesla Stock Price Prediction with ARIMA')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
# Define and train the SARIMA model
model_sarima = SARIMAX(train_data, 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 12))
model_sarima_fit = model_sarima.fit(disp=False)

# Forecast using SARIMA
forecast_sarima = model_sarima_fit.forecast(steps=len(test_data))
test_data['SARIMA_Prediction'] = forecast_sarima

# Plot the SARIMA predictions
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['TSLA'], label='Training Data')
plt.plot(test_data.index, test_data['TSLA'], label='Actual Price')
plt.plot(test_data.index, test_data['SARIMA_Prediction'], label='SARIMA Prediction')
plt.title('Tesla Stock Price Prediction with SARIMA')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
from sklearn.preprocessing import MinMaxScaler

# Scale data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for LSTM (60 days lookback)
def create_dataset(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Split and reshape data
lookback = 60
X_train, y_train = create_dataset(scaled_data[:train_size], lookback)
X_test, y_test = create_dataset(scaled_data[train_size:], lookback)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32)
def evaluate_model(true, predicted, model_name="Model"):
    if len(true) == 0 or len(predicted) == 0:
        print(f"{model_name}: No data available for evaluation!")
        return
    
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mape = mean_absolute_percentage_error(true, predicted) * 100
    
    print(f"📊 {model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
# Check the lengths and ensure no NaNs are present
print("Data ready for evaluation:")
print("TSLA length:", len(test_data['TSLA']))
print("ARIMA_Prediction length:", len(test_data['ARIMA_Prediction']))
print("SARIMA_Prediction length:", len(test_data['SARIMA_Prediction']))
print("LSTM Prediction length:", len(predicted_stock_price))
# Handle any remaining NaNs by filling or dropping them
test_data['ARIMA_Prediction'].fillna(method='ffill', inplace=True)
test_data['SARIMA_Prediction'].fillna(method='ffill', inplace=True)

# Ensure predicted_stock_price is aligned with true values
predicted_stock_price = predicted_stock_price[-len(test_data['TSLA']):]
# Ensure the predicted values array matches the length of the true values
predicted_stock_price = predicted_stock_price[-len(test_data['TSLA']):]

# Validate the lengths before evaluation
print("TSLA length:", len(test_data['TSLA']))
print("LSTM Prediction length:", len(predicted_stock_price))
# Check the shape of test_data before cleaning
print("Initial test_data shape:", test_data.shape)

# Check for NaNs in the specific columns
print(test_data[['TSLA', 'ARIMA_Prediction', 'SARIMA_Prediction']].isnull().sum())

# Fill NaNs in ARIMA and SARIMA prediction columns using forward fill
test_data['ARIMA_Prediction'].fillna(method='ffill', inplace=True)
test_data['SARIMA_Prediction'].fillna(method='ffill', inplace=True)

# Alternatively, fill NaNs with the mean of each column
# test_data['ARIMA_Prediction'].fillna(test_data['ARIMA_Prediction'].mean(), inplace=True)
# test_data['SARIMA_Prediction'].fillna(test_data['SARIMA_Prediction'].mean(), inplace=True)

# Check for NaNs after filling
print(test_data[['TSLA', 'ARIMA_Prediction', 'SARIMA_Prediction']].isnull().sum())
# Drop rows with NaNs only in 'TSLA' while keeping the predictions intact
test_data = test_data.dropna(subset=['TSLA'])

# Check the updated lengths of all relevant columns
print("After filling NaNs:")
print("Length of TSLA values:", len(test_data['TSLA']))
print("Length of ARIMA predictions:", len(test_data['ARIMA_Prediction']))
print("Length of SARIMA predictions:", len(test_data['SARIMA_Prediction']))
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def evaluate_model(true, predicted, model_name="Model"):
    # Check if arrays are empty
    if len(true) == 0 or len(predicted) == 0:
        print(f"{model_name}: No data available for evaluation!")
        return
    
    # Check for NaNs in the data
    if np.isnan(true).any() or np.isnan(predicted).any():
        print(f"{model_name}: Data contains NaN values! Evaluation aborted.")
        return
    
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mape = mean_absolute_percentage_error(true, predicted) * 100
    
    print(f"📊 {model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
print("Length of TSLA values:", len(test_data['TSLA']))
print("Length of LSTM predictions:", len(predicted_stock_price))
# Ensure the predicted values array matches the length of the true values
predicted_stock_price = predicted_stock_price[-len(test_data['TSLA']):]

# Recheck lengths to confirm alignment
print("Aligned TSLA length:", len(test_data['TSLA']))
print("Aligned LSTM Prediction length:", len(predicted_stock_price))
# Align 'TSLA' true values with the length of LSTM predictions
test_data = test_data[-len(predicted_stock_price):]

# Check the updated lengths
print("Final TSLA length:", len(test_data['TSLA']))
print("Final LSTM Prediction length:", len(predicted_stock_price))
def evaluate_model(true, predicted, model_name="Model"):
    if len(true) == 0 or len(predicted) == 0:
        print(f"{model_name}: No data available for evaluation!")
        return
    
    if np.isnan(true).any() or np.isnan(predicted).any():
        print(f"{model_name}: Data contains NaN values! Evaluation aborted.")
        return
    
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mape = mean_absolute_percentage_error(true, predicted) * 100
    
    print(f"📊 {model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

# Safely evaluate the LSTM model
evaluate_model(test_data['TSLA'], predicted_stock_price, "LSTM")
