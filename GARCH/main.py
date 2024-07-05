# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
from datetime import timedelta

# Load the uploaded CSV file
file_path = '/Users/admin/Documents/Projects/Visualisation/Data/3mpricehistory.csv'
data = pd.read_csv(file_path)

# Parse 'DateTime' as dates and set it as the index
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

# Assuming the CSV has a 'Index Price' column for prices
# Calculate returns (12-hour periods)
returns = 100 * data['Index Price'].pct_change().dropna()

# Visualize returns
plt.figure(figsize=(10,4))
plt.plot(returns)
plt.ylabel('Pct Return', fontsize=16)
plt.title('Returns', fontsize=20)
plt.show()

# Plot PACF
plot_pacf(returns**2)
plt.show()

# Fit GARCH(3,3) model
model_garch_3_3 = arch_model(returns, p=3, q=3)
model_fit_garch_3_3 = model_garch_3_3.fit()
print(model_fit_garch_3_3.summary())

# Fit GARCH(3,0) model (ARCH(3))
model_arch_3 = arch_model(returns, p=3, q=0)
model_fit_arch_3 = model_arch_3.fit()
print(model_fit_arch_3.summary())

# Rolling forecast with GARCH(3,0)
rolling_predictions = []
test_size = 365 * 2  # Adjust this based on your data (since data is in 12-hour periods, we double the days)

for i in range(test_size):
    train = returns.iloc[:-(test_size-i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-test_size:])

# Plot rolling forecast
plt.figure(figsize=(10,4))
true, = plt.plot(returns[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)
plt.show()

# Volatility prediction for next 7 days
train = returns
model = arch_model(train, p=2, q=2)
model_fit = model.fit(disp='off')
pred = model_fit.forecast(horizon=14)  # 7 days but 12-hour periods
future_dates = [returns.index[-1] + timedelta(hours=12*i) for i in range(1, 15)]
pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

# Plot next 7 days volatility prediction
plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Volatility Prediction - Next 7 Days (12-Hour Periods)', fontsize=20)
plt.show()