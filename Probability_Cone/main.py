import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the price history and 3-month implied volatility data from the uploaded CSV files
price_history_path = '/Users/admin/Documents/Projects/Visualisation/Data/pprrice.csv'
iv_data_path = '/Users/admin/Documents/Projects/Visualisation/Data/iviviviv.csv'
# Read the files
price_history_data = pd.read_csv(price_history_path)
iv_data = pd.read_csv(iv_data_path)

# Convert the 'DateTime' columns to datetime type for both datasets
price_history_data['DateTime'] = pd.to_datetime(price_history_data['DateTime'])
iv_data['DateTime'] = pd.to_datetime(iv_data['DateTime'])

# Merge the datasets on 'DateTime'
merged_data = pd.merge_asof(price_history_data.sort_values('DateTime'), iv_data.sort_values('DateTime'), on='DateTime')

# Calculate daily IV from 6-month IV (assuming 182 days in six months)
merged_data['Daily IV'] = merged_data['6M IV'] / 100 / np.sqrt(182)  # Adjusted for six months days

# Function to identify the six-month period start
def six_month_period(date):
    if date.month <= 6:
        return pd.Timestamp(year=date.year, month=1, day=1)
    else:
        return pd.Timestamp(year=date.year, month=7, day=1)

# Apply function to define six-month periods
merged_data['SixMonthStart'] = merged_data['DateTime'].apply(six_month_period)
six_month_starts = merged_data.groupby('SixMonthStart').first().reset_index()

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(merged_data['DateTime'], merged_data['Index Price'], label='Historical Price', color='gray')

# Constants for confidence intervals
ci_levels = [0.675, 1]  # Multipliers for 50% CI and 68% CI
ci_labels = ['50% CI', '68% CI']

# Generate and plot probability cones for each six-month period and confidence interval
for _, row in six_month_starts.iterrows():
    start_date = row['DateTime']
    start_price = row['Index Price']
    daily_iv = row['Daily IV']
    days = 182  # Days in six months
    time_points_future = np.arange(1, days + 1)
    for ci, label in zip(ci_levels, ci_labels):
        upper_prices_future = start_price * np.exp(daily_iv * ci * np.sqrt(time_points_future))
        lower_prices_future = start_price * np.exp(-daily_iv * ci * np.sqrt(time_points_future))
        future_dates = [start_date + pd.Timedelta(days=int(d - 1)) for d in time_points_future]
        plt.plot(future_dates, upper_prices_future, label=f'Upper Bound ({label}, Start: {start_date.date()})', linestyle='--')
        plt.plot(future_dates, lower_prices_future, label=f'Lower Bound ({label}, Start: {start_date.date()})', linestyle='--')
        plt.fill_between(future_dates, lower_prices_future, upper_prices_future, alpha=0.1)

plt.title('Price and Biannual Probability Cones for Different Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

plt.show()