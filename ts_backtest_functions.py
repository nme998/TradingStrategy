import numpy as np

class BacktestFunctions:
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