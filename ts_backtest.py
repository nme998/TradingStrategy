import numpy as np


class Trade:
    def __init__(self, entry_price, size, direction, stop_loss, take_profit, entry_date):
        self.entry_price = entry_price
        self.size = size
        self.direction = direction
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_date = entry_date

        self.is_open = True
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0

        self.holding_days = 0


class BacktestEngine:

    def __init__(self, initial_capital=10000, risk_per_trade=0.015, hmm_model=None, down_state=None, up_state=None):

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_total_risk = 0.1

        self.hmm_model = hmm_model
        self.down_state = down_state
        self.up_state = up_state

        self.open_trades = []
        self.closed_trades = []

        self.equity_curve = []
        self.dates = []
        self.returns_history = []
        self.prev_price = None

        # --- PARAMETERS ---
        self.entry_threshold = 1.0   
        self.exit_threshold = 0.5
        self.min_hold_days = 1
        self.max_hold_days = 5

        # --- DATA BUFFERS ---
        self.signal_history = []
        self.price_history = []

        # --- DEBUG ---
        self.stats = {
            "entries_total": 0,
            "entries_long": 0,
            "entries_short": 0,
            "entries_skipped_threshold": 0,
            "entries_skipped_consistency": 0,
            "entries_skipped_trend": 0,

            "exits_stop_loss": 0,
            "exits_take_profit": 0,
            "exits_signal": 0,
            "exits_time": 0
        }

    # -------------------------------
    # SIGNAL PIPELINE
    # -------------------------------

    def compute_raw_signal(self, prediction):
        return 0.5 * prediction[0] + 0.3 * prediction[1] + 0.2 * prediction[2]

    def normalize_signal(self, signal):
        self.signal_history.append(signal)

        if len(self.signal_history) < 20:
            return 0

        std = np.std(self.signal_history[-20:])
        if std == 0:
            return 0

        return signal / std
    
    def is_compressed(self):
        if len(self.price_history) < 30:
            return False

        recent_range = max(self.price_history[-10:]) - min(self.price_history[-10:])
        past_range = max(self.price_history[-30:]) - min(self.price_history[-30:])

        return recent_range < 0.5 * past_range
    
    def get_range_high_low(self):
        if len(self.price_history) < 15:
            return None, None

        window = self.price_history[-10:]
        return max(window), min(window)
    
    def get_atr(self, high, low):

        if len(self.price_history) < 20:
            return 0

        trs = []

        for i in range(1, len(self.price_history)):
            prev_close = self.price_history[i - 1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )

            trs.append(tr)

        return np.mean(trs[-14:])

    def compute_confidence(self, prediction):
        signs = np.sign(prediction)
        agreement = abs(np.sum(signs)) / 3  # 0 → no agreement, 1 → full

        magnitude = np.mean(np.abs(prediction))

        return agreement * magnitude

    def is_consistent(self, prediction):
        return (
            (prediction[0] > 0 or prediction[1] > 0 and prediction[2] > 0)
            or
            (prediction[0] < 0 or prediction[1] < 0 and prediction[2] < 0)
        )

    def trend_filter(self, price):
        self.price_history.append(price)

        if len(self.price_history) < 20:
            return 0  # no trend yet

        ma = np.mean(self.price_history[-20:])

        if price > ma:
            return 1
        elif price < ma:
            return -1
        else:
            return 0
        
    def get_current_regime(self):

        if self.hmm_model is None or len(self.returns_history) < 20:
            return None

        returns_window = np.array(self.returns_history[-50:]).reshape(-1, 1)

        hidden_states = self.hmm_model.predict(returns_window)
        current_state = hidden_states[-1]

        if current_state == self.up_state:
            return "high_vol"
        else:
            return "low_vol"
        
    def current_total_risk(self):

        total = 0

        for trade in self.open_trades:
            risk_per_share = abs(trade.entry_price - trade.stop_loss)
            trade_risk = risk_per_share * trade.size
            total += trade_risk

        return total / self.capital

    # -------------------------------
    # POSITION SIZING
    # -------------------------------

    def calculate_position_size(self, price, stop_loss_price, confidence):

        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(price - stop_loss_price)

        if risk_per_share == 0:
            return 0

        base_size = risk_amount / risk_per_share

        confidence = min(confidence * 10, 1.0)  # scale it properly

        return base_size * confidence

    # -------------------------------
    # TRADE MANAGEMENT
    # -------------------------------

    def open_trade(self, price, atr,  signal, confidence, date):

        direction = 1 if signal > 0 else -1

        if direction == 1:
            stop_loss = price - 1.5 * atr
            take_profit = price + 2.5 * atr
        else:
            stop_loss = price + 1.5 * atr
            take_profit = price - 2.5 * atr
        
        size = self.calculate_position_size(price, stop_loss, confidence)

        if size <= 0:
            return

        trade = Trade(price, size, direction, stop_loss, take_profit, date)
        self.open_trades.append(trade)

    def close_trade(self, trade, price, date, reason):

        trade.is_open = False
        trade.exit_price = price
        trade.exit_date = date

        trade.pnl = (
            (price - trade.entry_price)
            * trade.size
            * trade.direction
        )

        self.capital += trade.pnl
        self.closed_trades.append(trade)

        print(f"[{reason}] Entry {trade.entry_price:.2f} → Exit {price:.2f} | PnL {trade.pnl:.2f}")

    # -------------------------------
    # UPDATE TRADES
    # -------------------------------

    def update_trades(self, price, signal, date):

        for trade in self.open_trades[:]:

            trade.holding_days += 1

            # Stop loss
            if trade.direction == 1 and price <= trade.stop_loss:
                self.stats["exits_stop_loss"] += 1
                self.close_trade(trade, price, date, "SL")
                self.open_trades.remove(trade)
                continue

            if trade.direction == -1 and price >= trade.stop_loss:
                self.stats["exits_stop_loss"] += 1
                self.close_trade(trade, price, date, "SL")
                self.open_trades.remove(trade)
                continue

            # Take profit
            if trade.direction == 1 and price >= trade.take_profit:
                self.stats["exits_take_profit"] += 1
                self.close_trade(trade, price, date, "TP")
                self.open_trades.remove(trade)
                continue

            if trade.direction == -1 and price <= trade.take_profit:
                self.stats["exits_take_profit"] += 1
                self.close_trade(trade, price, date, "TP")
                self.open_trades.remove(trade)
                continue

            if trade.holding_days >= self.min_hold_days:

                trade_score = signal * trade.direction

                # --- HARD REVERSAL ---
                if trade_score < -self.exit_threshold:
                    self.stats["exits_signal"] += 1
                    self.close_trade(trade, price, date, "REV")
                    self.open_trades.remove(trade)
                    continue

                # --- WEAK SIGNAL ---
                if abs(signal) < self.exit_threshold * 0.5:
                    self.stats["exits_signal"] += 1
                    self.close_trade(trade, price, date, "WEAK")
                    self.open_trades.remove(trade)
                    continue

                # --- PROFIT PROTECTION ---
                unrealized_pnl = (
                    (price - trade.entry_price)
                    * trade.size
                    * trade.direction
                )

                if unrealized_pnl > 0:
                    if abs(signal) < self.exit_threshold:
                        self.stats["exits_signal"] += 1
                        self.close_trade(trade, price, date, "PROTECT")
                        self.open_trades.remove(trade)
                        continue

    # -------------------------------
    # STEP FUNCTION
    # -------------------------------

    def step(self, price, high, low, prediction, date):
        atr = self.get_atr(high, low)
        # -------------------------------
        # SIGNAL PIPELINE
        # -------------------------------
        raw_signal = self.compute_raw_signal(prediction)
        signal = self.normalize_signal(raw_signal)
        confidence = self.compute_confidence(prediction)
        trend = self.trend_filter(price)

        # -------------------------------
        # UPDATE EXISTING TRADES
        # -------------------------------
        self.update_trades(price, signal, date)

        # -------------------------------
        # UPDATE RETURNS (for HMM)
        # -------------------------------
        if self.prev_price is not None:
            ret = np.log(price / self.prev_price)
            self.returns_history.append(ret)

        self.prev_price = price

        # -------------------------------
        # GET REGIME
        # -------------------------------
        regime = self.get_current_regime()

        if regime is None:
            self.update_equity(price, date)
            return

        # -------------------------------
        # MOMENTUM STRATEGY (HIGH VOL)
        # -------------------------------
        if regime == "high_vol":

            threshold = self.entry_threshold * 0.8

            # Signal strength
            if abs(signal) < threshold:
                self.stats["entries_skipped_threshold"] += 1
                self.update_equity(price, date)
                return

            # Consistency check
            if not self.is_consistent(prediction):
                self.stats["entries_skipped_consistency"] += 1
                self.update_equity(price, date)
                return

            # Trend filter (only here)
            if trend != 0 and ((signal > 0 and trend != 1) or (signal < 0 and trend != -1)):
                self.stats["entries_skipped_trend"] += 1
                self.update_equity(price, date)
                return

            # Risk control
            if self.current_total_risk() < self.max_total_risk:

                self.open_trade(price, atr, signal, confidence, date)

                if abs(signal) > self.entry_threshold * 1.5:
                    self.open_trade(price, atr, signal, confidence, date)

                self.stats["entries_total"] += 1
                if signal > 0:
                    self.stats["entries_long"] += 1
                else:
                    self.stats["entries_short"] += 1

        # -------------------------------
        # MEAN REVERSION STRATEGY (LOW VOL)
        # -------------------------------
        
        elif regime == "low_vol":

            high_range, low_range = self.get_range_high_low()

            if high_range is None:
                self.update_equity(price, date)
                return

            compressed = self.is_compressed()

            # Only trade when market is compressed
            if not compressed:
                self.update_equity(price, date)
                return

            breakout_buffer = 0.2 * atr

            # --- BREAKOUT LONG ---
            if price > high_range + breakout_buffer and signal > 0:

                if self.current_total_risk() < self.max_total_risk:
                    self.open_trade(price, atr, signal, confidence, date)
                    self.stats["entries_total"] += 1
                    self.stats["entries_long"] += 1

            # --- BREAKOUT SHORT ---
            elif price < low_range - breakout_buffer and signal < 0:

                if self.current_total_risk() < self.max_total_risk:
                    self.open_trade(price, atr, signal, confidence, date)
                    self.stats["entries_total"] += 1
                    self.stats["entries_short"] += 1

            else:
                self.update_equity(price, date)
                return    

        # -------------------------------
        # UPDATE EQUITY
        # -------------------------------
        self.update_equity(price, date)

    # -------------------------------
    # EQUITY
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
    # DEBUG
    # -------------------------------

    def print_stats(self):
        print("\n=== TRADE DEBUG STATS ===")
        for k, v in self.stats.items():
            print(f"{k}: {v}")