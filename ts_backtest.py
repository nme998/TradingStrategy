import numpy as np
from ts_reversal_detection import detect_reversal
from ts_backtest_functions import  BacktestFunctions
from ts_strategies.ts_main_strat import MainStrat
from ts_strategies.ts_stat_arb import StatArb
from ts_strategies.ts_options_vol import OptionsVolatility
from ts_trade_manager import Trade, TradeManager
from ts_trade_manager_options import OptionTradeManager

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

        self.strategy = strategy
        self.option_trade_manager = OptionTradeManager(self)
        self.trade_manager = TradeManager(self)
        self.strategy.option_trade_manager = self.option_trade_manager
        self.strategy.trade_manager = self.trade_manager

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.portfolio_risk = 0.1
        self.portfolio_delta = 0
        self.portfolio_gamma = 0
        self.portfolio_theta = 0
        self.portfolio_vega = 0
        self.portfolio_option_value = 0

        self.hmm_models = hmm_models
        self.down_states = down_states  
        self.up_states = up_states

        self.lstm_models = lstm_models
        self.lstm_scalers = lstm_scalers

        self.open_trades = {}
        self.closed_trades = {}
        self.open_options_trades = {}
        self.closed_options_trades = {}
        self.open_hedges = {}
        self.closed_hedges = {}

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

    def step(self, date, rows, predictions, lookback_windows, vol_predictions=None, stat_arb_lookbacks=None):
        ticker_contexts = {}
        for ticker, row in rows.items():

            price = row["Close"]
            high = row["High"]
            low = row["Low"]
            prediction = predictions[ticker]
            vol_prediction = vol_predictions[ticker] if vol_predictions is not None else None
            lookback_window = lookback_windows[ticker]
            #print("LOOKBACK COLUMNS:", lookback_window.columns.tolist())

            if ticker not in self.prev_price:
                self.prev_price[ticker] = None

            if ticker not in self.returns_history:
                self.returns_history[ticker] = []

            if ticker not in self.price_history:
                self.price_history[ticker] = []
                self.high_history[ticker] = []
                self.low_history[ticker] = []

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
                
            if isinstance(self.strategy, OptionsVolatility):
                self.strategy.update_trades(engine = self, ticker = ticker, row = row, volatility_prediction = vol_prediction, regime = regime, lookback_windows = lookback_windows)

        if isinstance(self.strategy, StatArb):
            self.strategy.update_trades(engine = self, ticker_contexts = ticker_contexts, current_date = date, stat_arb_lookbacks = stat_arb_lookbacks)

        
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
        if isinstance(self.strategy, MainStrat): #and date.month in (5, 6, 7):
            self.strategy.process_entries(engine=self, rows=rows, ticker_contexts=ticker_contexts)
        elif isinstance(self.strategy, StatArb):
            self.strategy.process_entries(engine=self, rows=rows, ticker_contexts=ticker_contexts, current_date=date, stat_arb_lookbacks=stat_arb_lookbacks)
        elif isinstance(self.strategy, OptionsVolatility):
            self.strategy.process_entries(engine=self, rows=rows, volatility_predictions=vol_predictions, return_predictions=predictions, regime = regime, lookback_windows=lookback_windows)

        self.update_equity(date)

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