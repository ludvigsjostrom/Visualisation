# Cryptocurrency Analysis Tools

This repository contains three Python scripts for analyzing cryptocurrency price data using different statistical and financial models.

## Contents

1. [GARCH Model Analysis](#garch-model-analysis)
2. [Jump-Diffusion Model](#jump-diffusion-model)
3. [Probability Cone Analysis](#probability-cone-analysis)

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- arch
- statsmodels

You can install the required packages using:

```
pip install pandas numpy matplotlib arch statsmodels
```

## GARCH Model Analysis

File: `GARCH/main.py`

This script performs GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model analysis on cryptocurrency price data.

Features:
- Loads and preprocesses price data
- Calculates returns
- Fits GARCH(3,3) and GARCH(3,0) models
- Performs rolling forecast
- Predicts volatility for the next 7 days

## Jump-Diffusion Model

File: `Jump-Diffusion/main.py`

This script implements a Jump-Diffusion model to simulate future price paths for a cryptocurrency.

Features:
- Loads historical price data
- Simulates future price paths using a Jump-Diffusion model
- Visualizes historical data and simulated future paths

## Probability Cone Analysis

File: `Probability_Cone/main.py`

This script creates probability cones for cryptocurrency price predictions based on historical data and implied volatility.

Features:
- Loads price history and implied volatility data
- Calculates daily implied volatility
- Generates probability cones for different confidence intervals
- Visualizes historical prices and probability cones

## Usage

1. Ensure you have the required packages installed.
2. Update the file paths in each script to point to your data files.
3. Run each script individually:

```
python GARCH/main.py
python Jump-Diffusion/main.py
python Probability_Cone/main.py
```

## Data

The scripts expect CSV files containing cryptocurrency price data and volatility information. Make sure to update the file paths in each script to point to your data files.
