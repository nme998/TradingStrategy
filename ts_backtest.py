import numpy as np
from ts_reversal_detection import detect_reversal
from ts_backtest_functions import  BacktestFunctions
from ts_strategies.ts_main_strat import MainStrat
from ts_strategies.ts_stat_arb import StatArb


class Trade:
    next_id = 0

    def __init__(self, entry_price, size, direction, stop_loss, take_profit, entry_date, strategy, type):
        self.trade_id = Trade.next_id
        Trade.next_id += 1
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

        self.prediction = None
        self.signal = None
        self.confidence = None
        self.regime = None
        self.entropy = None
        self.trend = None

class TradeContext:
    def __init__(self, prediction, raw_signal, signal, confidence, regime, entropy, trend, atr, lookback_window, date):

        self.prediction = prediction
        self.raw_signal = raw_signal
        self.signal = signal
        self.confidence = confidence
        self.regime = regime
        self.entropy = entropy
        self.trend = trend
        self.atr = atr
        self.lookback_window = lookback_window
        self.date = date

class BacktestEngine(BacktestFunctions):

    def __init__(self, initial_capital=10000, risk_per_trade=0.015, 
                 hmm_models=None, down_states=None, up_states=None, 
                 lstm_models=None, lstm_scalers=None, strategy=None):

        self.strategy = strategy or MainStrat()
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
    # TRADE MANAGEMENT
    # -------------------------------

    def open_trade(self, ticker, fill_price, size, take_profit, stop_loss, score, type, context):

        direction = 1 if context.signal > 0 else -1
        slippage = context.atr * self.slippage_factor
        if size <= 0:
            return

        trade = Trade(fill_price, size, direction, stop_loss, take_profit, context.date, context.regime, type)
        if ticker not in self.open_trades:
            self.open_trades[ticker] = []
        self.open_trades[ticker].append(trade)

        trade.entry_slippage = slippage
        trade.prediction = context.prediction
        trade.signal = context.signal
        trade.confidence = context.confidence
        trade.regime = context.regime
        trade.entropy = context.entropy
        trade.trend = context.trend
        trade.score = score

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

    def update_trades(self, ticker, price, context):

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


                exit_score = self.strategy.compute_exit_score(self.exit_threshold, trade, ticker, price, context.signal, context.lookback_window)

                if exit_score >= 3:
                    self.stats["exits_score"] += 1
                    self.close_trade(ticker, trade, price, context.date, "SCORE_EXIT")
                    self.open_trades[ticker].remove(trade)
                    continue

                if trade.type == "long":
                    # Stop loss
                    if price <= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, context.date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue
                    
                    # Take profit
                    if price >= trade.take_profit:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, context.date, "TP")
                        self.open_trades[ticker].remove(trade)
                        continue
                    
                elif trade.type == "short":
                    if price >= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, context.date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue

                    if price <= trade.take_profit:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, context.date, "TP")
                        self.open_trades[ticker].remove(trade)
                        continue

            elif trade.strategy == "low_vol":
                unrealized_pnl = ((price - trade.entry_price) * trade.size * trade.direction)
                if unrealized_pnl > 0:
                    if abs(context.signal) < self.exit_threshold * 0.5:
                        self.stats["exits_take_profit"] += 1
                        self.close_trade(ticker, trade, price, context.date, "PROTECT")
                        self.open_trades[ticker].remove(trade)
                        continue
                
                if trade.type == "long":
                    # Stop loss
                    if price <= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, context.date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue

                elif trade.type == "short":
                    if price >= trade.stop_loss:
                        self.stats["exits_stop_loss"] += 1
                        self.close_trade(ticker, trade, price, context.date, "SL")
                        self.open_trades[ticker].remove(trade)
                        continue
                    
            if trade.holding_days >= self.min_hold_days:
                trade_score = context.signal * trade.direction
                # --- HARD REVERSAL ---
                if trade_score < -self.exit_threshold:
                    self.stats["exits_signal"] += 1
                    self.close_trade(ticker, trade, price, context.date, "REV")
                    self.open_trades[ticker].remove(trade)
                    continue
    # -------------------------------
    # STEP FUNCTION
    # -------------------------------
    def step(self, date, rows, predictions, lookback_windows):

        ticker_contexts = {}

        # ===================================
        # BUILD CONTEXT FOR EVERY TICKER
        # ===================================

        for ticker, row in rows.items():

            price = row["Close"]
            high = row["High"]
            low = row["Low"]
            prediction = predictions[ticker]
            lookback_window = lookback_windows[ticker]

            # -------------------------------
            # INIT STORAGE
            # -------------------------------

            if ticker not in self.prev_price:
                self.prev_price[ticker] = None

            if ticker not in self.returns_history:
                self.returns_history[ticker] = []

            if ticker not in self.price_history:
                self.price_history[ticker] = []
                self.high_history[ticker] = []
                self.low_history[ticker] = []

            # -------------------------------
            # ATR
            # -------------------------------

            self.price_history[ticker].append(price)
            self.high_history[ticker].append(high)
            self.low_history[ticker].append(low)

            atr = self.get_atr(ticker)
            raw_signal = self.compute_raw_signal(prediction)
            signal = self.normalize_signal(ticker, raw_signal)
            regime = self.get_current_regime(ticker)
            entropy = self.calculate_permutation_entropy(lookback_window["return"], dx=3)
            trend = self.trend_filter(ticker, price)
            confidence = self.compute_confidence(ticker, prediction, signal, regime,  price, lookback_window)
            
            context = TradeContext(prediction=prediction, raw_signal=raw_signal, signal=signal, confidence=confidence,
                regime=regime, entropy=entropy, trend=trend, atr=atr, lookback_window=lookback_window,  date=date)

            ticker_contexts[ticker] = context

        # ===================================
        # UPDATE OPEN TRADES
        # ===================================

        for ticker, row in rows.items():

            
            if isinstance(self.strategy, MainStrat):
                self.strategy.update_trades(engine = self, ticker = ticker, price = row["Close"], context = ticker_contexts[ticker])
                #self.update_trades(ticker, row["Close"], ticker_contexts[ticker])
            elif isinstance(self.strategy, StatArb):
                self.strategy.update_trades(
                    self,
                    ticker,
                    row["Close"],
                    ticker_contexts[ticker]
                )

        # ===================================
        # UPDATE RETURNS
        # ===================================

        for ticker, row in rows.items():

            price = row["Close"]

            if self.prev_price[ticker] is not None:
                ret = np.log(price / self.prev_price[ticker])
                self.returns_history[ticker].append(ret)

            self.prev_price[ticker] = price
            self.current_prices[ticker] = price

        # ===================================
        # ENTRY LOGIC
        # ===================================

        if isinstance(self.strategy, MainStrat):
            self.strategy.process_entries(engine=self, rows=rows, ticker_contexts=ticker_contexts)
        elif isinstance(self.strategy, StatArb):
            self.strategy.process_entries(engine=self, rows=rows, ticker_contexts=ticker_contexts, current_date=date)

        # ===================================
        # EQUITY
        # ===================================

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
                unrealized += ((current_price - trade.entry_price) * trade.size * trade.direction)

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