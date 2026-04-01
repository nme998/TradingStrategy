from ts_walkforward import run_walkforward_backtest
from ts_metrics import PerformanceMetrics
import pandas as pd
import numpy as np

ticker = "TSLA"

engine, data, all_predictions = run_walkforward_backtest(ticker)

# -------------------- PERFORMANCE METRICS --------------------

# Convert equity curve → pandas Series
equity_series = pd.Series(engine.equity_curve)

metrics = PerformanceMetrics(equity_series)

# Extract trade PnLs
trade_pnls = [t.pnl for t in engine.closed_trades]

print("\n=== PERFORMANCE METRICS ===")
print("Sharpe Ratio:", metrics.sharpe_ratio())
print("Max Drawdown:", metrics.max_drawdown())
print("Win Rate:", metrics.win_rate(trade_pnls))
print("Profit Factor:", metrics.profit_factor(trade_pnls))
print("Expectancy:", metrics.expectancy(trade_pnls))

# -------------------- DEBUG SECTION --------------------

print("\n=== DEBUG INFO ===")

print("Total Trades:", len(engine.closed_trades))

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
    print("Trades per step:", len(engine.closed_trades) / len(engine.equity_curve))

engine.print_stats()

print("\nBacktest complete!")