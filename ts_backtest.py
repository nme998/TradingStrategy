import numpy as np


class Trade:
    def __init__(self, entry_price, size, direction, stop_loss, take_profit, entry_date):
        self.entry_price = entry_price
        self.size = size
        self.direction = direction  # 1 = long, -1 = short
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_date = entry_date

        self.is_open = True
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0

        self.holding_days = 0


class BacktestEngine:

    def __init__(self, initial_capital=10000, risk_per_trade=0.01):

        self.initial_capital = initial_capital
        self.capital = initial_capital

        self.risk_per_trade = risk_per_trade

        self.open_trades = []
        self.closed_trades = []

        self.equity_curve = []
        self.dates = []

        # --- STRATEGY PARAMS ---
        self.entry_threshold = 0.005
        self.exit_threshold = 0.002
        self.min_hold_days = 1

        # --- DEBUG STATS ---
        self.stats = {
            "entries_total": 0,
            "entries_long": 0,
            "entries_short": 0,
            "entries_skipped_threshold": 0,
            "entries_skipped_consistency": 0,

            "exits_stop_loss": 0,
            "exits_take_profit": 0,
            "exits_signal": 0
        }

    # -------------------------------
    # SIGNAL GENERATION
    # -------------------------------
    def compute_signal(self, prediction):
        return (
            0.5 * prediction[0] +
            0.3 * prediction[1] +
            0.2 * prediction[2]
        )

    def is_consistent(self, prediction):
        return (
            (prediction[0] > 0 and prediction[1] > 0 and prediction[2] > 0)
            or
            (prediction[0] < 0 and prediction[1] < 0 and prediction[2] < 0)
        )

    # -------------------------------
    # POSITION SIZING
    # -------------------------------
    def calculate_position_size(self, price, stop_loss_price, signal):

        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(price - stop_loss_price)

        if risk_per_share == 0:
            return 0

        base_size = risk_amount / risk_per_share

        # Confidence scaling
        confidence = min(abs(signal) / 0.02, 1.0)
        print(f"Calculated position size: {base_size:.2f} shares (confidence: {confidence:.2f})")

        return base_size * confidence

    # -------------------------------
    # OPEN TRADE
    # -------------------------------
    def open_trade(self, price, signal, date):

        direction = 1 if signal > 0 else -1

        reward_pct = 0.04  # 2:1 RR

        if direction == 1:
            stop_loss = price * (1 - 0.035)
            take_profit = price * (1 + reward_pct)
        else:
            stop_loss = price * (1 + 0.035)
            take_profit = price * (1 - reward_pct)

        size = self.calculate_position_size(price, stop_loss, signal)

        if size <= 0:
            return

        trade = Trade(price, size, direction, stop_loss, take_profit, date)
        self.open_trades.append(trade)

    # -------------------------------
    # CLOSE TRADE
    # -------------------------------
    def close_trade(self, trade, price, date):

        trade.is_open = False
        trade.exit_price = price
        trade.exit_date = date

        trade.pnl = (
            (price - trade.entry_price)
            * trade.size
            * trade.direction
        )
        print(f"Closing trade: Entry {trade.entry_price:.2f}, Exit {price:.2f}, PnL {trade.pnl:.2f}")

        self.capital += trade.pnl
        self.closed_trades.append(trade)

    # -------------------------------
    # UPDATE OPEN TRADES
    # -------------------------------
    def update_trades(self, price, prediction, date):

        signal = self.compute_signal(prediction)

        for trade in self.open_trades[:]:

            trade.holding_days += 1

            # --- Stop-loss ---
            if trade.direction == 1 and price <= trade.stop_loss:
                self.stats["exits_stop_loss"] += 1
                self.close_trade(trade, price, date)
                self.open_trades.remove(trade)
                continue

            if trade.direction == -1 and price >= trade.stop_loss:
                self.stats["exits_stop_loss"] += 1
                self.close_trade(trade, price, date)
                self.open_trades.remove(trade)
                continue

            # --- Take-profit ---
            if trade.direction == 1 and price >= trade.take_profit:
                self.stats["exits_take_profit"] += 1
                self.close_trade(trade, price, date)
                self.open_trades.remove(trade)
                continue

            if trade.direction == -1 and price <= trade.take_profit:
                self.stats["exits_take_profit"] += 1
                self.close_trade(trade, price, date)
                self.open_trades.remove(trade)
                continue

            # --- Minimum hold ---
            if trade.holding_days < self.min_hold_days:
                continue

            # --- Signal exit ---
            if trade.direction == 1 and signal < -self.exit_threshold:
                self.stats["exits_signal"] += 1
                self.close_trade(trade, price, date)
                self.open_trades.remove(trade)
                continue

            if trade.direction == -1 and signal > self.exit_threshold:
                self.stats["exits_signal"] += 1
                self.close_trade(trade, price, date)
                self.open_trades.remove(trade)
                continue

    # -------------------------------
    # STEP FUNCTION
    # -------------------------------
    def step(self, price, prediction, date):

        # 1. Update trades
        self.update_trades(price, prediction, date)

        # 2. Compute signal
        signal = self.compute_signal(prediction)
        consistent = self.is_consistent(prediction)

        # 3. Entry logic with stats
        if abs(signal) <= self.entry_threshold:
            self.stats["entries_skipped_threshold"] += 1

        elif not consistent:
            self.stats["entries_skipped_consistency"] += 1

        elif len(self.open_trades) == 0:
            self.open_trade(price, signal, date)

            self.stats["entries_total"] += 1

            if signal > 0:
                self.stats["entries_long"] += 1
            else:
                self.stats["entries_short"] += 1

        # 4. Track equity
        self.update_equity(price, date)

    # -------------------------------
    # EQUITY TRACKING
    # -------------------------------
    def update_equity(self, current_price, date):

        unrealized = 0

        for trade in self.open_trades:
            unrealized += (
                (current_price - trade.entry_price)
                * trade.size
                * trade.direction
            )

        total_equity = self.capital + unrealized

        self.equity_curve.append(total_equity)
        self.dates.append(date)

    # -------------------------------
    # PRINT DEBUG STATS
    # -------------------------------
    def print_stats(self):
        print("\n=== TRADE DEBUG STATS ===")
        for k, v in self.stats.items():
            print(f"{k}: {v}")