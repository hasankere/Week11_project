import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio():
    df = pd.read_csv('historical_data.csv', index_col='Date', parse_dates=True)
    returns = df.pct_change().dropna()
    
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = ((0,1), (0,1), (0,1))
    initial_weights = [1/3, 1/3, 1/3]
    
    result = minimize(sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return {"optimized_weights": result.x.tolist()}
