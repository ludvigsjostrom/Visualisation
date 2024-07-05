import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file
file_path = '/Users/admin/Documents/Projects/Visualisation/Data/BTC Volatility Chart.csv'
btc_data = pd.read_csv(file_path)

# Convert 'DateTime' to datetime type and filter the last year's data
btc_data['DateTime'] = pd.to_datetime(btc_data['DateTime'])
last_year_data = btc_data[btc_data['DateTime'] >= '2023-05-15']

# Get the last price from the data to start the probability cone
last_price = last_year_data['Index Price'].iloc[-1]

# Constants for probability cone calculation
annual_iv = 0.645  # Annual Implied Volatility
ci_values = [0.5, 1]  # Standard deviations for 50% and 68% CI
colors = ['orange', 'green']  # Colors for different CIs
labels = ['50% CI', '68% CI']  # Labels for the plots
days = 365  # Number of days in the projection
daily_iv_crypto = annual_iv / np.sqrt(365)  # Daily volatility for cryptocurrency

# Parameters for jump diffusion
lambda_j = 22 / 365  # Daily jump intensity (number of jumps per year converted to daily rate)
mu_j = -0.00978  # Mean of the log of jump sizes
sigma_j = 0.11846  # Standard deviation of the log of jump sizes

# Time points for future projection
time_points_future = np.arange(1, days + 1)

# Simulate future paths with jump diffusion
np.random.seed(0)  # For reproducibility
jump_sizes = np.random.lognormal(mean=mu_j, sigma=sigma_j, size=(days, 1000)) - 1
jumps = np.random.poisson(lam=lambda_j, size=(days, 1000))
daily_returns = daily_iv_crypto * np.random.randn(days, 1000) + jumps * jump_sizes
future_prices = last_price * np.exp(np.cumsum(daily_returns, axis=0))

# Plot historical data and future probability cones
plt.figure(figsize=(14, 8))
plt.plot(last_year_data['DateTime'], last_year_data['Index Price'], label='Historical Price', color='black')

# Plot simulated paths for different confidence intervals
for i in range(1000):  # Plot each simulated path
    plt.plot(last_year_data['DateTime'].iloc[-1] + pd.to_timedelta(time_points_future - 1, unit='D'), future_prices[:, i], color='grey', alpha=0.05)

plt.title('Time Series Future Probability Cones with Jump Diffusion Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
