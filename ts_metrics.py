import numpy as np
import pandas as pd

class PerformanceMetrics:
    def __init__(self, equity_curve):
        """
        equity_curve: pandas Series of portfolio value over time
        """
        # Handle BacktestEngine input
        if hasattr(equity_curve, "equity_curve"):
            equity_curve = equity_curve.equity_curve

        # Convert to pandas Series if needed
        if not isinstance(equity_curve, pd.Series):
            equity_curve = pd.Series(equity_curve)

        self.equity_curve = equity_curve
        self.returns = self.equity_curve.pct_change().fillna(0)

    def sharpe_ratio(self, risk_free_rate=0.0):
        """
        Annualized Sharpe Ratio
        """
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        if std_return == 0:
            return np.nan
        return (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)

    def max_drawdown(self):
        """
        Maximum drawdown of the equity curve
        """
        cumulative_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cumulative_max) / cumulative_max
        return drawdown.min()

    def win_rate(self, trades):
        """
        trades: list of trade PnL (profit/loss)
        """
        wins = [t for t in trades if t > 0]
        return len(wins) / len(trades) if trades else np.nan

    def profit_factor(self, trades):
        """
        Profit factor = sum(profits) / sum(losses)
        """
        profits = sum(t for t in trades if t > 0)
        losses = -sum(t for t in trades if t < 0)
        return profits / losses if losses != 0 else np.inf

    def expectancy(self, trades):
        """
        Expectancy = avg(win) * win rate - avg(loss) * loss rate
        """
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        n_total = len(trades)
        if n_total == 0:
            return np.nan
        win_rate = len(wins)/n_total
        loss_rate = len(losses)/n_total
        avg_win = np.mean(wins) if wins else 0
        avg_loss = -np.mean(losses) if losses else 0
        return (avg_win * win_rate) - (avg_loss * loss_rate)
    
#TODO: Add more metrics - Sortino ratio, Calmar ratio, alpha/beta, ValueatRisk, Volatility, Exposure.