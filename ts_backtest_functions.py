import numpy as np
from ordpy import permutation_entropy

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

    def current_total_risk(self):
        total = 0

        for ticker in self.open_trades:
            for trade in self.open_trades[ticker]:
                risk_per_share = abs(trade.entry_price - trade.stop_loss)
                trade_risk = risk_per_share * trade.size
                total += trade_risk

        return total / self.capital
    
    def calculate_permutation_entropy(self, returns, dx=3):
        entropy =  permutation_entropy(returns, dx=dx, normalized=True)
        return entropy
    
    def calculate_model_confidence(self, val_predictions, current_prediction):
        predictions = np.array([x["prediction"] for x in val_predictions])
        actuals = np.array([[x["actual_1"], x["actual_2"],  x["actual_3"]] for x in val_predictions])

        # Direction confidence
        predicted_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)

        direction_accuracy = (predicted_direction == actual_direction).mean()
        direction_confidence = direction_accuracy * 100


        # Magnitude confidence
        magnitude_error = np.abs(predictions - actuals)

        mean_magnitude_error = magnitude_error.mean()
        magnitude_confidence = max(0, min(100, (1 - mean_magnitude_error) * 100))


        return {
            "direction_confidence": direction_confidence,
            "magnitude_confidence": magnitude_confidence,
            "validation_direction_accuracy": direction_accuracy,
            "mean_prediction_error": mean_magnitude_error
        }