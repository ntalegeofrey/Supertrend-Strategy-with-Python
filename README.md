# Supertrend Trading Strategy

This repository contains Python code implementing a Supertrend trading strategy. The Supertrend is a trend-following indicator that helps identify trend direction and potential entry and exit points in the market.

## Overview

The code consists of several functions:

1. **fetch_asset_data**: Fetches historical OHLC (Open-High-Low-Close) data for a specified asset from a cryptocurrency exchange.
2. **supertrend**: Calculates the Supertrend indicator based on the input OHLC data and a specified ATR (Average True Range) multiplier.
3. **generate_signals**: Generates buy/sell signals based on Supertrend values.
4. **create_positions**: Creates buy/sell positions based on the generated signals.
5. **plot_data**: Plots the Supertrend indicator along with buy/sell positions on a candlestick chart.
6. **strategy_performance**: Evaluates the performance of the trading strategy, including overall profit/loss, maximum drawdown, and other metrics.
7. **plot_performance_curve**: Plots the performance curve of the strategy against a benchmark.

## Usage

1. Ensure you have the necessary dependencies installed (`ccxt`, `pandas`, `numpy`, `pandas_ta`, `mplfinance`).
2. Replace the `symbol`, `start_date`, `interval`, and `exchange` variables with your desired asset, start date, interval, and exchange.
3. Adjust the `volatility` parameter for the Supertrend calculation.
4. Run the script to fetch data, apply the Supertrend strategy, evaluate performance, and visualize results.

## Dependencies

- `ccxt`: For fetching cryptocurrency exchange data.
- `pandas`, `numpy`: For data manipulation and analysis.
- `pandas_ta`: For technical analysis indicators.
- `mplfinance`: For plotting candlestick charts.
- `matplotlib`: For additional plotting functionalities.

## Disclaimer

- This code is provided for educational purposes only. Trading involves risk, and past performance is not indicative of future results. Always perform thorough testing and analysis before implementing any trading strategy in live markets.

## Contact

For any questions or inquiries, please contact [your_email@example.com](mailto:your_email@example.com).
