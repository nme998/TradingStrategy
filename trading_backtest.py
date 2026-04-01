import subprocess
import pandas as pd
import numpy as np
from trading_model import model_run
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------------------
# 1 Run prediction script
# -------------------------------
print("Running ensemble prediction...")

# Returns both predictions and stock_data
predicted_returns, stock_data = model_run()
pred = predicted_returns[0]  # array of predicted returns for each step

print("Predictions shape:", np.array(predicted_returns).shape)

# -------------------------------
# 2 Trading strategy
# -------------------------------
initial_capital = 10000
cash = initial_capital
shares = 0

portfolio_values = []
daily_returns = []

position = 0

for i in range(len(predicted_returns)):

    price = stock_data["Close"].iloc[i]  # actual close price
    signal = predicted_returns[i]  # predicted return for this step

    # BUY signal
    if position == 0 and signal > 0:
        shares = cash // price
        cash -= shares * price
        position = 1

    # SELL signal
    elif position == 1 and signal < 0:
        cash += shares * price
        shares = 0
        position = 0

    portfolio_value = cash + shares * price
    portfolio_values.append(portfolio_value)

    if i > 0:
        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        daily_returns.append(daily_return)

# Close final position
if shares > 0:
    cash += stock_data["Close"].iloc[len(predicted_returns)-1]
final_value = cash

# -------------------------------
# 3 Metrics
# -------------------------------
daily_returns = np.array(daily_returns)

mean_return = np.mean(daily_returns)
std_return = np.std(daily_returns)

# Sharpe Ratio (annualized)
sharpe_ratio = 0 if std_return == 0 else (mean_return / std_return) * np.sqrt(252)

# Sortino Ratio (annualized)
downside_returns = daily_returns[daily_returns < 0]
downside_std = np.std(downside_returns)
sortino_ratio = 0 if downside_std == 0 else (mean_return / downside_std) * np.sqrt(252)

total_return = (final_value - initial_capital) / initial_capital

# -------------------------------
# 4 Results
# -------------------------------
print("\nBacktest Results")
print("---------------------")
print(f"Final Portfolio Value: {final_value:.2f}")
print(f"Total Return: {total_return*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Sortino Ratio: {sortino_ratio:.3f}")

# -------------------------------
# 5 Plot predicted prices (for human visualization)
# -------------------------------
def plot_predicted_prices(stock_data, predicted_returns):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    predicted_returns = np.array(predicted_returns).flatten()
    
    if len(predicted_returns) == 0:
        print("No predicted returns to plot!")
        return

    # Start from last known close (or first close if you want full series)
    predicted_prices = [stock_data["Close"].iloc[0]]

    for r in predicted_returns:
        predicted_prices.append(predicted_prices[-1] * (1 + r))

    # Now predicted_prices length = len(predicted_returns)+1
    predicted_prices = np.array(predicted_prices)

    # Make x-axis match predicted prices
    predicted_index = stock_data.index[:len(predicted_prices)]

    plt.figure(figsize=(14, 6))
    plt.plot(stock_data.index, stock_data["Close"], label="Actual Close", color="black")
    plt.plot(predicted_index, predicted_prices, label="Predicted Prices", color="red", alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_predicted_prices(stock_data, predicted_returns)