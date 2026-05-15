import numpy as np
from ts_reversal_detection import detect_reversal


class Trade:
    def __init__(self, entry_price, size, direction, stop_loss, take_profit, entry_date, strategy, type):
        self.entry_price = entry_price
        self.size = size
        self.direction = direction
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_date = entry_date
        self.strategy = strategy
        self.type = type
        self.score = 0

        self.is_open = True
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0

        self.entry_slippage = 0
        self.exit_slippage = 0
        self.transaction_cost = 0
        self.pyramid = 0

        self.holding_days = 0


class BacktestEngine:

    def __init__(self, initial_capital=10000, risk_per_trade=0.015, 
                 hmm_models=None, down_states=None, up_states=None, 
                 lstm_models=None, lstm_scalers=None):

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_total_risk = 0.1

        self.hmm_models = hmm_models
        self.down_states = down_states
        self.up_states = up_states

        self.lstm_models = lstm_models
        self.lstm_scalers = lstm_scalers

        self.open_trades = {}
        self.closed_trades = {}

        self.equity_curve = []
        self.dates = []
        self.returns_history = {}
        self.high_history = {}
        self.low_history = {}
        self.current_prices = {}
        self.prev_price = {}
        self.signal_percentiles = {}

        # --- PARAMETERS ---
        self.entry_threshold = 1.0   
        self.exit_threshold = 0.5
        self.min_hold_days = 1
        self.max_hold_days = 5
        self.slippage_factor = 0.02
        self.transaction_cost_pct = 0.001

        # --- DATA BUFFERS ---
        self.signal_history = {}
        self.price_history = {}

        # --- DEBUG ---
        self.stats = {
            "entries_total": 0,
            "momentum_entries": 0,
            "breakout_entries": 0,
            "high_vol_entries": 0,
            "low_vol_entries": 0,
            "entries_long": 0,
            "entries_short": 0,
            "entries_skipped_threshold": 0,
            "entries_skipped_consistency": 0,
            "entries_skipped_trend": 0,
            "entries_skipped_score": 0,

            "exits_stop_loss": 0,
            "exits_take_profit": 0,
            "exits_signal": 0,
            "exits_score": 0
        }

    # -------------------------------
    # SIGNAL PIPELINE
    # -------------------------------

    def compute_raw_signal(self, prediction):
        return 0.5 * prediction[0] + 0.3 * prediction[1] + 0.2 * prediction[2]

    def normalize_signal(self, ticker, signal):
        if ticker not in self.signal_history:
            self.signal_history[ticker] = []

        self.signal_history[ticker].append(signal)

        if len(self.signal_history[ticker]) < 20:
            return 0

        std = np.std(self.signal_history[ticker][-20:])
        if std == 0:
            return 0

        return signal / std
    
    def is_compressed(self, ticker):

        if ticker not in self.price_history or len(self.price_history[ticker]) < 30:
            return False

        prices = self.price_history[ticker]

        recent_range = max(prices[-10:]) - min(prices[-10:])
        past_range = max(prices[-30:]) - min(prices[-30:])

        return recent_range < 0.5 * past_range


    def get_range_high_low(self, ticker):

        if ticker not in self.price_history or len(self.price_history[ticker]) < 15:
            return None, None

        window = self.price_history[ticker][-10:]
        return max(window), min(window)
    
    def get_atr(self, ticker):
        if ticker not in self.price_history:
            return 0

        closes = self.price_history[ticker]
        highs = self.high_history[ticker]
        lows = self.low_history[ticker]

        if len(closes) < 20:
            return 0

        trs = []

        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            trs.append(tr)

        return np.mean(trs[-14:])
    
    def calc_Z_score(self, data):
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0
        
        return (data[-1] - mean) / std
    
    def compute_confidence(self, ticker, prediction, signal, regime, price, lookback_window):

        ema20 = float(lookback_window["EMA_20"].iloc[-1])
        ema50 = float(lookback_window["EMA_50"].iloc[-1])
        rsi = float(lookback_window["RSI"].iloc[-1])
        macd = float(lookback_window["MACD"].iloc[-1])
        vwap = float(np.mean(lookback_window["Close"].iloc[-20:]))
        raw_pred_strength = np.mean(np.abs(prediction))


        if ticker not in self.signal_percentiles:
            self.signal_percentiles[ticker] = []
        self.signal_percentiles[ticker].append(raw_pred_strength)
        window = min(len(self.signal_percentiles[ticker]), 100)
        history = self.signal_percentiles[ticker][-window:]
        
        percentile = (
            sum(x <= raw_pred_strength for x in history)
            / len(history)
        )

        model_score = percentile


        regime_score = 1.0 if regime == "high_vol" else 0.4

        trend_score = 0

        if ema20 is not None and ema50 is not None:
            if signal > 0 and ema20 > ema50:
                trend_score += 1

            elif signal < 0 and ema20 < ema50:
                trend_score += 1


        structure_score = 0

        if signal > 0 and price > vwap:
            structure_score += 1

        elif signal < 0 and price < vwap:
            structure_score += 1


        pullback_score = 0

        if ema20:

            distance_to_ema = abs(price - ema20) / price

            # close to EMA = better pullback
            pullback_score = max(0, 1 - (distance_to_ema * 50))


        rsi_score = 0

        if signal > 0 and 40 <= rsi <= 60:
            rsi_score = 1

        elif signal < 0 and 40 <= rsi <= 60:
            rsi_score = 1


        macd_score = 0

        if signal > 0 and macd > 0:
            macd_score = 1

        elif signal < 0 and macd < 0:
            macd_score = 1

        confidence = (
            0.35 * model_score +
            0.20 * regime_score +
            0.15 * trend_score +
            0.10 * structure_score +
            0.10 * pullback_score +
            0.05 * rsi_score +
            0.05 * macd_score
        )

        return min(max(confidence, 0), 1)

    def is_consistent(self, prediction):
        return (
            (prediction[0] > 0 or prediction[1] > 0 and prediction[2] > 0)
            or
            (prediction[0] < 0 or prediction[1] < 0 and prediction[2] < 0)
        )

    def trend_filter(self, ticker, price):

        if len(self.price_history[ticker]) < 20:
            return 0

        ma = np.mean(self.price_history[ticker][-20:])

        if price > ma:
            return 1
        elif price < ma:
            return -1
        else:
            return 0
        
    def get_current_regime(self, ticker):
        if self.hmm_models[ticker] is None:
            return None

        if ticker not in self.returns_history:
            return None

        if len(self.returns_history[ticker]) < 20:
            return None

        returns_window = np.array(self.returns_history[ticker][-50:]).reshape(-1, 1)

        hidden_states = self.hmm_models[ticker].predict(returns_window)
        current_state = hidden_states[-1]

        return "high_vol" if current_state == self.up_states[ticker] else "low_vol"

    def compute_entry_score(self, ticker, prediction, signal, regime, price, lookback_window):

        score = 0
        ema20 = float(lookback_window["EMA_20"].iloc[-1])
        ema50 = float(lookback_window["EMA_50"].iloc[-1])
        rsi = float(lookback_window["RSI"].iloc[-1])
        macd = float(lookback_window["MACD"].iloc[-1])
        vwap = float(np.mean(lookback_window["Close"].iloc[-20:]))

        if signal > 0:
            score += 3
        elif signal < 0:
            score += 3

        if regime == "high_vol":
            score += 2


        if signal > 0 and ema20 > ema50:
            score += 1
        elif signal < 0 and ema20 < ema50:
            score += 1


        if signal > 0 and price > vwap:
            score += 1
        elif signal < 0 and price < vwap:
            score += 1

        distance = abs(price - ema20) / price

        pullback_score = max(0, 1 - distance * 10)

        score += 2 * pullback_score


        #if 40 <= rsi <= 55:
        #    score += 1


        if signal > 0 and macd > 0:
            score += 1
        elif signal < 0 and macd < 0:
            score += 1

        return score
    
    def compute_exit_score(self, trade, ticker, price, signal, lookback_window):

        score = 0
        ema20 = float(lookback_window["EMA_20"].iloc[-1])
        ema50 = float(lookback_window["EMA_50"].iloc[-1])
        rsi = float(lookback_window["RSI"].iloc[-1])
        macd = float(lookback_window["MACD"].iloc[-1])
        vwap = float(np.mean(lookback_window["Close"].iloc[-20:]))

        if abs(signal) < self.exit_threshold:
            score += 2  # strong exit signal


        if trade.type == "long" and ema20 < ema50:
            score += 2

        if trade.type == "short" and ema20 > ema50:
            score += 2


        if trade.type == "long" and price < vwap:
            score += 1

        if trade.type == "short" and price > vwap:
            score += 1

        unrealized_pnl = (price - trade.entry_price) * trade.size * trade.direction

        if unrealized_pnl < 0:
            score += 1  

        return score
    
    def score_to_multiplier(self, score):

        if score <= 4:
            return 0.3   # small size
        elif score <= 6:
            return 0.7   # medium size
        else:
            return 1.5   # full size
        
    def get_volatility(self, ticker):

        prices = self.price_history[ticker]

        if len(prices) < 20:
            return 1

        returns = np.diff(np.log(prices[-20:]))

        return np.std(returns)

    def calculate_position_size(self, price, stop_loss_price, score, ticker):

        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(price - stop_loss_price)

        if risk_per_share == 0:
            return 0

        base_size = risk_amount / risk_per_share

        # =========================
        # SCORE MULTIPLIER
        # =========================
        score_multiplier = self.score_to_multiplier(score)

        # =========================
        # VOLATILITY NORMALISATION
        # =========================
        vol = self.get_volatility(ticker)

        target_vol = 0.01  # tuning parameter

        vol_adjustment = target_vol / (vol + 1e-8)

        # =========================
        # FINAL SIZE
        # =========================
        size = base_size * score_multiplier * vol_adjustment

        return size

    def current_total_risk(self):
        total = 0

        for ticker in self.open_trades:
            for trade in self.open_trades[ticker]:
                risk_per_share = abs(trade.entry_price - trade.stop_loss)
                trade_risk = risk_per_share * trade.size
                total += trade_risk

        return total / self.capital

    # -------------------------------
    # TRADE MANAGEMENT
    # -------------------------------

    def open_trade(self, ticker, price, atr,  signal, confidence, date, strategy, type, score):

        direction = 1 if signal > 0 else -1

        slippage = atr * self.slippage_factor

        if type == "long":
            fill_price = price + slippage
            if strategy == "high_vol":
                stop_loss = fill_price - 1.5 * atr
                take_profit = fill_price + 2.5 * atr
            elif strategy == "low_vol":
                stop_loss = fill_price - 0.1 * atr
                take_profit = fill_price + atr
        elif type == "short":
            fill_price = price - slippage
            if strategy == "high_vol":
                stop_loss = fill_price + 1.5 * atr
                take_profit = fill_price - 2.5 * atr
            elif strategy == "low_vol":
                stop_loss = fill_price + 0.1 * atr
                take_profit = fill_price - atr

        size = self.calculate_position_size(fill_price, stop_loss, score, ticker)

        if size <= 0:
            return

        trade = Trade(fill_price, size, direction, stop_loss, take_profit, date, strategy, type)
        if ticker not in self.open_trades:
            self.open_trades[ticker] = []
        self.open_trades[ticker].append(trade)

        trade.entry_slippage = slippage

        side = "BUY" if direction == 1 else "SELL"
        #print(f"{side} Signal: {ticker} |  Date: {date} | Price: {price:.2f} | Size: {size:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Confidence: {confidence:.2f}")

    def close_trade(self, ticker, trade, price, date, reason):

        atr = self.get_atr(ticker)

        slippage = atr * self.slippage_factor

        if trade.direction == 1:
            exit_fill = price - slippage
        else:
            exit_fill = price + slippage

        trade_value = (
            abs(trade.entry_price * trade.size)
            +
            abs(exit_fill * trade.size)
        )

        transaction_cost = (
            trade_value * self.transaction_cost_pct
        )

        trade.is_open = False
        trade.exit_price = exit_fill
        trade.exit_date = date
        trade.exit_slippage = slippage
        trade.transaction_cost = transaction_cost

        gross_pnl = (
            (exit_fill - trade.entry_price)
            * trade.size
            * trade.direction
        )

        trade.pnl = gross_pnl - transaction_cost

        self.capital += trade.pnl

        if ticker not in self.closed_trades:
            self.closed_trades[ticker] = []

        self.closed_trades[ticker].append(trade)

        '''
        print(
            f"[{ticker}] [{reason}] "
            f"EntryDate ({trade.entry_date}) "
            f"Entry {trade.entry_price:.2f} → "
            f"ExitDate ({trade.exit_date}) "
            f"Exit {price:.2f} | "
            f"PnL {trade.pnl:.2f}"
        )
        '''
    # -------------------------------
    # UPDATE TRADES
    # -------------------------------

    def update_trades(self, ticker, price, signal, date, confidence, score, lookback_window):

        if ticker not in self.open_trades:
            return

        for trade in self.open_trades[ticker][:]:

            trade.holding_days += 1

            if trade.strategy == "high_vol":
                atr = self.get_atr(ticker)

                if trade.type == "long":
                    new_stop = price - 1.5 * atr
                    trade.stop_loss = max(trade.stop_loss, new_stop)

                elif trade.type == "short":
                    new_stop = price + 1.5 * atr
                    trade.stop_loss = min(trade.stop_loss, new_stop)

                unrealized_pnl = ((price - trade.entry_price) * trade.size * trade.direction)
                if unrealized_pnl > 0:
                    if abs(signal) > self.entry_threshold * 1.5 and trade.pyramid > 2:
                        self.stats["entries_total"] += 1
                        self.stats["momentum_entries"] += 1
                        trade.pyramid += 1

                        if signal > 0:
                            self.open_trade(ticker, price, atr, signal, confidence, date, trade.strategy, "long", score)
                            self.stats["entries_long"] += 1
                        else:
                            self.open_trade(ticker, price, atr, signal, confidence, date, trade.strategy, "short", score)
                            self.stats["entries_short"] += 1

                    if abs(signal) < self.exit_threshold:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, date, "PROTECT")
                        self.open_trades[ticker].remove(trade)
                        continue
                
                exit_score = self.compute_exit_score(trade, ticker, price, signal, lookback_window)

                if exit_score >= 3:
                    self.stats["exits_score"] += 1
                    self.close_trade(ticker, trade, price, date, "SCORE_EXIT")
                    self.open_trades[ticker].remove(trade)
                    continue

                if trade.type == "long":
                    # Stop loss
                    if price <= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue
                    
                    # Take profit
                    if price >= trade.take_profit:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, date, "TP")
                        self.open_trades[ticker].remove(trade)
                        continue
                    
                elif trade.type == "short":
                    if price >= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue

                    if price <= trade.take_profit:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, date, "TP")
                        self.open_trades[ticker].remove(trade)
                        continue

            elif trade.strategy == "low_vol":
                unrealized_pnl = ((price - trade.entry_price) * trade.size * trade.direction)
                if unrealized_pnl > 0:
                    if abs(signal) < self.exit_threshold * 0.5:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, date, "PROTECT")
                        self.open_trades[ticker].remove(trade)
                        continue
                
                if trade.type == "long":
                    # Stop loss
                    if price <= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue
                    '''
                    # Take profit
                    if price >= trade.take_profit:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, date, "TP")
                        self.open_trades[ticker].remove(trade)
                        continue
                    '''
                elif trade.type == "short":
                    if price >= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue
                    
                    '''
                    if price <= trade.take_profit:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, date, "TP")
                        self.open_trades[ticker].remove(trade)
                        continue
                    '''
            if trade.holding_days >= self.min_hold_days:

                trade_score = signal * trade.direction
                
                # --- HARD REVERSAL ---
                if trade_score < -self.exit_threshold:
                    self.stats["exits_signal"] += 1
                    self.close_trade(ticker, trade, price, date, "REV")
                    self.open_trades[ticker].remove(trade)
                    continue
                '''
                # --- WEAK SIGNAL ---
                if abs(signal) < self.exit_threshold * 0.5:
                    self.stats["exits_signal"] += 1
                    self.close_trade(ticker, trade, price, date, "WEAK")
                    self.open_trades[ticker].remove(trade)
                    continue
                '''
    # -------------------------------
    # STEP FUNCTION
    # -------------------------------
    def step(self, ticker, price, high, low, prediction, date, lookback_window=None):
        # -------------------------------
        # INIT STORAGE
        # -------------------------------
        if ticker not in self.prev_price:
            self.prev_price[ticker] = None

        if ticker not in self.returns_history:
            self.returns_history[ticker] = []

        # -------------------------------
        # ATR
        # -------------------------------
        if ticker not in self.price_history:
            self.price_history[ticker] = []
            self.high_history[ticker] = []
            self.low_history[ticker] = []

        self.price_history[ticker].append(price)
        self.high_history[ticker].append(high)
        self.low_history[ticker].append(low)
        atr = self.get_atr(ticker)

        # -------------------------------
        # SIGNAL PIPELINE
        # -------------------------------
        raw_signal = self.compute_raw_signal(prediction)
        signal = self.normalize_signal(ticker, raw_signal)
        regime = self.get_current_regime(ticker)
        trend = self.trend_filter(ticker, price)
        confidence = self.compute_confidence(ticker, prediction, signal, regime, price, lookback_window)
        score = self.compute_entry_score(ticker, prediction, signal, regime, price, lookback_window)
        
        
        #reversal = detect_reversal(self.lstm_models[ticker], self.lstm_scalers[ticker], lookback_window)
        #if reversal > 0.7:
        #    print(f"Reversal Probability: {reversal:.2f}"f" at Date ({date}) "f"for Ticker ({ticker})")

        self.update_trades(ticker, price, signal, date, confidence, score, lookback_window)

        if self.prev_price[ticker] is not None:
            ret = np.log(price / self.prev_price[ticker])
            self.returns_history[ticker].append(ret)

        self.prev_price[ticker] = price
        self.current_prices[ticker] = price

        if regime is None:
            self.update_equity(date)
            return

        # ===============================
        # HIGH VOL → MOMENTUM
        # ===============================
        if regime == "high_vol":

            if score < 6:
                self.stats["entries_skipped_score"] += 1
                self.update_equity(date)
                return

            threshold = self.entry_threshold * 0.8
            self.stats["high_vol_entries"] += 1

            if abs(signal) < threshold:
                self.stats["entries_skipped_threshold"] += 1
                self.update_equity(date)
                return

            if not self.is_consistent(prediction):
                self.stats["entries_skipped_consistency"] += 1
                self.update_equity(date)
                return

            if trend != 0 and ((signal > 0 and trend != 1) or (signal < 0 and trend != -1)):
                self.stats["entries_skipped_trend"] += 1
                self.update_equity(date)
                return

            if self.current_total_risk() < self.max_total_risk:
                self.stats["entries_total"] += 1
                self.stats["momentum_entries"] += 1
                if signal > 0:
                    self.open_trade(ticker, price, atr, signal, confidence, date, regime, "long", score)
                    self.stats["entries_long"] += 1
                else:
                    self.open_trade(ticker, price, atr, signal, confidence, date, regime, "short", score)
                    self.stats["entries_short"] += 1
                
                if abs(signal) > self.entry_threshold * 1.5:
                    self.stats["entries_total"] += 1
                    self.stats["momentum_entries"] += 1
                    if signal > 0:
                        self.open_trade(ticker, price, atr, signal, confidence, date, regime, "long", score)
                        self.stats["entries_long"] += 1
                    else:
                        self.open_trade(ticker, price, atr, signal, confidence, date, regime, "short", score)
                        self.stats["entries_short"] += 1
                    
                
        # ===============================
        # LOW VOL → BREAKOUT
        # ===============================
        elif regime == "low_vol":
            self.stats["low_vol_entries"] += 1
                    

        # -------------------------------
        # EQUITY UPDATE
        # -------------------------------
        self.update_equity(date)

    # -------------------------------
    # EQUITY
    # -------------------------------

    def update_equity(self, date):
        unrealized = 0

        for ticker in self.open_trades:
            if ticker not in self.current_prices:
                continue

            current_price = self.current_prices[ticker]

            for trade in self.open_trades[ticker]:
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

#TODO: Add pair trading strategy