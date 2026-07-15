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

    def sharpe_ratio(self, risk_free_rate=0.03):
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
    
    def CAGR(self):
        """
        Compound Annual Growth Rate
        """
        total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        n_years = len(self.equity_curve) / 252  
        if n_years == 0:
            return np.nan
        return (1 + total_return) ** (1/n_years) - 1

    def sortino_ratio(self, risk_free_rate=0.03):
        """
        Annualized Sortino Ratio
        """
        excess_returns = self.returns - (risk_free_rate / 252)

        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        if downside_std == 0 or np.isnan(downside_std):
            return np.nan

        return excess_returns.mean() / downside_std * np.sqrt(252)


    def value_at_risk(self, confidence=0.95):
        """
        Historical Value at Risk (VaR)

        Returns the expected one-day loss at the chosen confidence level.
        Example:
            confidence=0.95 -> 5th percentile
        """
        percentile = (1 - confidence) * 100
        return np.percentile(self.returns, percentile)


    def calmar_ratio(self):
        """
        Annualized Calmar Ratio
        """
        cagr = self.CAGR()
        max_dd = abs(self.max_drawdown())

        if max_dd == 0:
            return np.nan

        return cagr / max_dd
    
#TODO: Add more metrics - Sortino ratio, Calmar ratio, alpha/beta, ValueatRisk, Volatility, Exposure.