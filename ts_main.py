from ts_walkforward import run_walkforward_backtest
from ts_metrics import PerformanceMetrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import chain

tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "AMD", "GOOGL", "META"]
#tickers = ["JPM", "BAC", "WFC", "C", "GS", "MS"]

engine, all_predictions = run_walkforward_backtest(tickers)

trade_pnls = [
    t.pnl
    for ticker in engine.closed_trades
    for t in engine.closed_trades[ticker]
]

def plot_equity_and_returns(engine):
    equity = np.array(engine.equity_curve)
    returns = np.diff(equity) / equity[:-1]

    plt.figure(figsize=(10, 5))

    plt.plot(engine.dates, equity, label="Equity")

    plt.plot(engine.dates[1:], returns * equity.max(), label="Returns", color="orange")

    plt.title("Equity Curve + Returns")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------- PERFORMANCE METRICS --------------------

# Convert equity curve → pandas Series
equity_series = pd.Series(engine.equity_curve)

metrics = PerformanceMetrics(equity_series)

# Extract trade PnLs
trade_pnls = [
    trade.pnl
    for trades in chain(engine.closed_trades.values(), engine.closed_options_trades.values())
    for trade in trades
]

print("\n=== PERFORMANCE METRICS ===")
print("Sharpe Ratio:", metrics.sharpe_ratio())
print("Sortino:", metrics.sortino_ratio())
print("Max Drawdown:", metrics.max_drawdown())
print("Win Rate:", metrics.win_rate(trade_pnls))
print("Profit Factor:", metrics.profit_factor(trade_pnls))
print("Expectancy:", metrics.expectancy(trade_pnls))
print("CAGR:", metrics.CAGR())
print("Calmar:", metrics.calmar_ratio())
print("VaR (95%):", metrics.value_at_risk())
print("CVaR (99%):", metrics.value_at_risk(confidence=0.99))

# -------------------- DEBUG SECTION --------------------
print("\n=== DEBUG INFO ===")

total_trades = sum(
    len(trades)
    for trades in chain(engine.closed_trades.values(), engine.closed_options_trades.values())
)
open_trades = sum(
    len(trades)
    for trades in chain(engine.open_trades.values(), engine.open_options_trades.values())
)

print("Total Trades:", total_trades)
print("Open Trades:", open_trades)

if trade_pnls:
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]

    print("Avg Win:", np.mean(wins) if wins else 0)
    print("Avg Loss:", np.mean(losses) if losses else 0)
    print("Max Win:", max(trade_pnls))
    print("Max Loss:", min(trade_pnls))

print("Start Equity:", engine.equity_curve[0] if engine.equity_curve else 0)
print("End Equity:", engine.equity_curve[-1] if engine.equity_curve else 0)

print("Final Capital:", engine.capital)

if len(engine.equity_curve) > 0:
    print("Trades per step:", total_trades / len(engine.equity_curve))

engine.print_stats()

rows = []

for ticker, trades in chain(engine.closed_trades.items(), engine.closed_options_trades.items()):
    for trade in trades:
        row = trade.__dict__.copy()
        row["ticker"] = ticker
        rows.append(row)

df = pd.DataFrame(rows)

df.to_csv("tradebook.csv", index=False)

print("\nBacktest complete!") 

plot_equity_and_returns(engine)

#TODO: Add function to select stocks for trading.