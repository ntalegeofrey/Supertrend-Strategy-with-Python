import ccxt
import warnings
from matplotlib.pyplot import fill_between
import pandas as pd
import numpy as np
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def fetch_asset_data(symbol, start_date, interval, exchange):
    # Convert start_date to milliseconds timestamp
    start_date_ms = exchange.parse8601(start_date)
    ohlcv = exchange.fetch_ohlcv(symbol, interval, since=start_date_ms)
    header = ["date", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=header)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index("date", inplace=True)
    # Drop the last row containing live data
    df.drop(df.index[-1], inplace=True)
    return df

def supertrend(df, atr_multiplier=3):
    # Calculate the Upper Band(UB) and the Lower Band(LB)
    # Formular: Supertrend =(High+Low)/2 + (Multiplier)∗(ATR)
    current_average_high_low = (df['high']+df['low'])/2
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], period=15)
    df.dropna(inplace=True)
    df['basicUpperband'] = current_average_high_low + (atr_multiplier * df['atr'])
    df['basicLowerband'] = current_average_high_low - (atr_multiplier * df['atr'])
    first_upperBand_value = df['basicUpperband'].iloc[0]
    first_lowerBand_value = df['basicLowerband'].iloc[0]
    upperBand = [first_upperBand_value]
    lowerBand = [first_lowerBand_value]

    for i in range(1, len(df)):
        if df['basicUpperband'].iloc[i] < upperBand[i-1] or df['close'].iloc[i-1] > upperBand[i-1]:
            upperBand.append(df['basicUpperband'].iloc[i])
        else:
            upperBand.append(upperBand[i-1])

        if df['basicLowerband'].iloc[i] > lowerBand[i-1] or df['close'].iloc[i-1] < lowerBand[i-1]:
            lowerBand.append(df['basicLowerband'].iloc[i])
        else:
            lowerBand.append(lowerBand[i-1])

    df['upperband'] = upperBand
    df['lowerband'] = lowerBand
    df.drop(['basicUpperband', 'basicLowerband',], axis=1, inplace=True)
    return df

def generate_signals(df):
    # Intiate a signals list
    signals = [0]

    # Loop through the dataframe
    for i in range(1 , len(df)):
        if df['close'][i] > df['upperband'][i]:
            signals.append(1)
        elif df['close'][i] < df['lowerband'][i]:
            signals.append(-1)
        else:
            signals.append(signals[i-1])

    # Add the signals list as a new column in the dataframe
    df['signals'] = signals
    df['signals'] = df["signals"].shift(1) #Remove look ahead bias
    return df

def create_positions(df):
    # We need to shut off (np.nan) data points in the upperband where the signal is not 1
    df['upperband'][df['signals'] == 1] = np.nan
    # We need to shut off (np.nan) data points in the lowerband where the signal is not -1
    df['lowerband'][df['signals'] == -1] = np.nan

    # Create a positions list
    buy_positions = [np.nan]
    sell_positions = [np.nan]

    # Loop through the dataframe
    for i in range(1, len(df)):
        # If the current signal is a 1 (Buy) & the it's not equal to the previous signal
        # Then that is a trend reversal, so we BUY at that current market price
        # We take note of the upperband value
        if df['signals'][i] == 1 and df['signals'][i] != df['signals'][i-1]:
            buy_positions.append(df['close'][i])
            sell_positions.append(np.nan)
        # If the current signal is a -1 (Sell) & the it's not equal to the previous signal
        # Then that is a trend reversal, so we SELL at that current market price
        elif df['signals'][i] == -1 and df['signals'][i] != df['signals'][i-1]:
            sell_positions.append(df['close'][i])
            buy_positions.append(np.nan)
        else:
            buy_positions.append(np.nan)
            sell_positions.append(np.nan)

    # Add the positions list as a new column in the dataframe
    df['buy_positions'] = buy_positions
    df['sell_positions'] = sell_positions
    return df

def plot_data(df, symbol):
    # Define lowerband line plot
    lowerband_line = mpf.make_addplot(df['lowerband'], label= "lowerband", color='green')
    # Define upperband line plot
    upperband_line = mpf.make_addplot(df['upperband'], label= "upperband", color='red')
    # Define buy and sell markers
    buy_position_makers = mpf.make_addplot(df['buy_positions'], type='scatter', marker='^', label= "Buy", markersize=80, color='#2cf651')
    sell_position_makers = mpf.make_addplot(df['sell_positions'], type='scatter', marker='v', label= "Sell", markersize=80, color='#f50100')
    # A list of all addplots(apd)
    apd = [lowerband_line, upperband_line, buy_position_makers, sell_position_makers]
    # Create fill plots
    lowerband_fill = dict(y1=df['close'].values, y2=df['lowerband'].values, panel=0, alpha=0.3, color="#CCFFCC")
    upperband_fill = dict(y1=df['close'].values, y2=df['upperband'].values, panel=0, alpha=0.3, color="#FFCCCC")
    fills = [lowerband_fill, upperband_fill]
    # Plot the data 
    mpf.plot(df, addplot=apd, type='candle', volume=True, style='charles', xrotation=20, title=str(symbol + ' Supertrend Plot'), fill_between=fills)

def strategy_performance(strategy_df, capital=100, leverage=1):
    # Initialize the performance variables
    cumulative_balance = capital
    investment = capital
    pl = 0
    max_drawdown = 0
    max_drawdown_percentage = 0

    # Lists to store intermediate values for calculating metrics
    balance_list = [capital]
    pnl_list = [0]
    investment_list = [capital]
    peak_balance = capital

    # Loop from the second row (index 1) of the DataFrame
    for index in range(1, len(strategy_df)):
        row = strategy_df.iloc[index]

        # Calculate P/L for each trade signal
        if row['signals'] == 1:
            pl = ((row['close'] - row['open']) / row['open']) * \
                investment * leverage
        elif row['signals'] == -1:
            pl = ((row['open'] - row['close']) / row['close']) * \
                investment * leverage
        else:
            pl = 0

        # Update the investment if there is a signal reversal
        if row['signals'] != strategy_df.iloc[index - 1]['signals']:
            investment = cumulative_balance

        # Calculate the new balance based on P/L and leverage
        cumulative_balance += pl

        # Update the investment list
        investment_list.append(investment)

        # Calculate the cumulative balance and add it to the DataFrame
        balance_list.append(cumulative_balance)

        # Calculate the overall P/L and add it to the DataFrame
        pnl_list.append(pl)

        # Calculate max drawdown
        drawdown = cumulative_balance - peak_balance
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            max_drawdown_percentage = (max_drawdown / peak_balance) * 100

        # Update the peak balance
        if cumulative_balance > peak_balance:
            peak_balance = cumulative_balance

    # Add new columns to the DataFrame
    strategy_df['investment'] = investment_list
    strategy_df['cumulative_balance'] = balance_list
    strategy_df['pl'] = pnl_list
    strategy_df['cumPL'] = strategy_df['pl'].cumsum()

    # Calculate other performance metrics (replace with your calculations)
    overall_pl_percentage = (
        strategy_df['cumulative_balance'].iloc[-1] - capital) * 100 / capital
    overall_pl = strategy_df['cumulative_balance'].iloc[-1] - capital
    min_balance = min(strategy_df['cumulative_balance'])
    max_balance = max(strategy_df['cumulative_balance'])

    # Print the performance metrics
    print("Overall P/L: {:.2f}%".format(overall_pl_percentage))
    print("Overall P/L: {:.2f}".format(overall_pl))
    print("Min balance: {:.2f}".format(min_balance))
    print("Max balance: {:.2f}".format(max_balance))
    print("Maximum Drawdown: {:.2f}".format(max_drawdown))
    print("Maximum Drawdown %: {:.2f}%".format(max_drawdown_percentage))

    # Return the Strategy DataFrame
    return strategy_df

# Plot the performance curve
def plot_performance_curve(strategy_df):
    plt.plot(strategy_df['cumulative_balance'], label='Strategy')
    plt.title('Performance Curve')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Initialize data fetch parameters
    symbol = "BTC/USDT"
    start_date = "2022-12-1"
    interval = '4h'
    exchange = ccxt.binance()

    # Fetch historical OHLC data for ETH/USDT
    data = fetch_asset_data(symbol=symbol, start_date=start_date, interval=interval, exchange=exchange)


    volatility = 3

    # Apply supertrend formula
    supertrend_data = supertrend(df=data, atr_multiplier=volatility)

    # Generate the Signals
    supertrend_positions = generate_signals(supertrend_data)

    # Generate the Positions
    supertrend_positions = create_positions(supertrend_positions)

    # Calculate performance
    supertrend_df = strategy_performance(supertrend_positions, capital=100, leverage=1)
    print(supertrend_df)

    # Plot data
    plot_data(supertrend_positions, symbol=symbol)
    
    # Plot the performance curve
    plot_performance_curve(supertrend_df)
