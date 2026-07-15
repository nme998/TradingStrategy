import numpy as np

class MainStrat:

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
    
    def compute_exit_score(self, exit_threshold, trade, ticker, price, signal, lookback_window):

        score = 0
        ema20 = float(lookback_window["EMA_20"].iloc[-1])
        ema50 = float(lookback_window["EMA_50"].iloc[-1])
        vwap = float(np.mean(lookback_window["Close"].iloc[-20:]))

        if abs(signal) < exit_threshold:
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
    
    def calculate_position_size(self, engine, price, stop_loss_price, score, ticker, entropy):

        risk_amount = engine.capital * engine.risk_per_trade
        risk_per_share = abs(price - stop_loss_price)

        if risk_per_share == 0:
            return 0

        base_size = risk_amount / risk_per_share

        # =========================
        # SCORE MULTIPLIER
        # =========================
        score_multiplier = engine.score_to_multiplier(score)

        # =========================
        # VOLATILITY NORMALISATION
        # =========================
        vol = engine.get_volatility(ticker)

        target_vol = 0.01  # tuning parameter

        vol_adjustment = target_vol / (vol + 1e-8)

        # =========================
        # FINAL SIZE
        # =========================
        size = base_size * score_multiplier * vol_adjustment

        if entropy is not None:
            if entropy < 0.9:
                size *= 1.5
            elif entropy < 0.93:
                size *= 1.7
            elif entropy < 0.95:
                size *= 0.5
            else:
                size *= 0.8

        return size
    
    def update_trades(self, engine, ticker, price, context):
        if ticker not in engine.open_trades:
            return

        for trade in engine.open_trades[ticker][:]:

            trade.holding_days += 1

            if trade.strategy == "high_vol":
                atr = engine.get_atr(ticker)

                if trade.type == "long":
                    new_stop = price - 1.5 * atr
                    trade.stop_loss = max(trade.stop_loss, new_stop)

                elif trade.type == "short":
                    new_stop = price + 1.5 * atr
                    trade.stop_loss = min(trade.stop_loss, new_stop)


                exit_score = self.compute_exit_score(engine.exit_threshold, trade, ticker, price, context.signal, context.lookback_window)

                if exit_score >= 3:
                    engine.stats["exits_score"] += 1
                    engine.trade_manager.close_trade(ticker, trade, price, context.date, "SCORE_EXIT")
                    engine.open_trades[ticker].remove(trade)
                    continue

                if trade.type == "long":
                    # Stop loss
                    if price <= trade.stop_loss:
                        engine.stats["exits_stop_loss"] += 1
                        engine.trade_manager.close_trade(ticker, trade, price, context.date, "SL")
                        engine.open_trades[ticker].remove(trade)
                        continue
                    
                    # Take profit
                    if price >= trade.take_profit:
                        engine.stats["exits_take_profit"] += 1
                        engine.trade_manager.close_trade(ticker, trade, price, context.date, "TP")
                        engine.open_trades[ticker].remove(trade)
                        continue
                    
                elif trade.type == "short":
                    if price >= trade.stop_loss:
                        engine.stats["exits_stop_loss"] += 1
                        engine.trade_manager.close_trade(ticker, trade, price, context.date, "SL")
                        engine.open_trades[ticker].remove(trade)
                        continue

                    if price <= trade.take_profit:
                        engine.stats["exits_take_profit"] += 1
                        engine.trade_manager.close_trade(ticker, trade, price, context.date, "TP")
                        engine.open_trades[ticker].remove(trade)
                        continue

            elif trade.strategy == "low_vol":
                unrealized_pnl = ((price - trade.entry_price) * trade.size * trade.direction)
                if unrealized_pnl > 0:
                    if abs(context.signal) < engine.exit_threshold * 0.5:
                        engine.stats["exits_take_profit"] += 1
                        engine.trade_manager.close_trade(ticker, trade, price, context.date, "PROTECT")
                        engine.open_trades[ticker].remove(trade)
                        continue
                
                if trade.type == "long":
                    # Stop loss
                    if price <= trade.stop_loss:
                        engine.stats["exits_stop_loss"] += 1
                        engine.trade_manager.close_trade(ticker, trade, price, context.date, "SL")
                        engine.open_trades[ticker].remove(trade)
                        continue

                elif trade.type == "short":
                    if price >= trade.stop_loss:
                        engine.stats["exits_stop_loss"] += 1
                        engine.trade_manager.close_trade(ticker, trade, price, context.date, "SL")
                        engine.open_trades[ticker].remove(trade)
                        continue
                    
            if trade.holding_days >= engine.min_hold_days:
                trade_score = context.signal * trade.direction
                # --- HARD REVERSAL ---
                if trade_score < -engine.exit_threshold:
                    engine.stats["exits_signal"] += 1
                    engine.trade_manager.close_trade(ticker, trade, price, context.date, "REV")
                    engine.open_trades[ticker].remove(trade)
                    continue

    def process_entries(self, engine, rows, ticker_contexts):

        for ticker, context in ticker_contexts.items():

            price = rows[ticker]["Close"]
            slippage = context.atr * engine.slippage_factor
            if context.signal > 0:
                fill_price = price + slippage
                stop_loss = fill_price - 1.5 * context.atr
                take_profit = fill_price + 2.5 * context.atr
            else:
                fill_price = price - slippage
                stop_loss = fill_price + 1.5 * context.atr
                take_profit = fill_price - 2.5 * context.atr

            score = self.compute_entry_score(ticker, context.prediction, context.signal, context.regime, price, context.lookback_window)
            size = self.calculate_position_size(engine, fill_price, stop_loss, score, ticker, entropy = context.entropy)

            if context.regime != "high_vol":
                continue

            if score < 6:
                engine.stats["entries_skipped_score"] += 1
                continue

            if context.confidence < 0.5:
                engine.stats["entries_skipped_score"] += 1
                continue

            threshold = engine.entry_threshold * 0.8

            engine.stats["high_vol_entries"] += 1

            if abs(context.signal) < threshold:
                engine.stats["entries_skipped_threshold"] += 1
                continue

            if not engine.is_consistent(
                context.prediction
            ):
                engine.stats["entries_skipped_consistency"] += 1
                continue

            if (
                context.trend != 0 and
                (
                    (context.signal > 0 and context.trend != 1)
                    or
                    (context.signal < 0 and context.trend != -1)
                )
            ):
                engine.stats["entries_skipped_trend"] += 1
                continue

            if (engine.current_total_risk() >= engine.max_total_risk):
                continue

            engine.stats["entries_total"] += 1
            engine.stats["momentum_entries"] += 1

            if context.signal > 0:
                engine.trade_manager.open_trade(ticker, fill_price, size, take_profit, stop_loss, score, "long", context)
                engine.stats["entries_long"] += 1

            else:
                engine.trade_manager.open_trade(ticker, fill_price, size, take_profit, stop_loss, score, "short", context)
                engine.stats["entries_short"] += 1

            # strong signal add-on

            if (abs(context.signal) > engine.entry_threshold * 1.5):
                engine.stats["entries_total"] += 1
                engine.stats["momentum_entries"] += 1

                if context.signal > 0:
                    engine.trade_manager.open_trade( ticker, fill_price, size, take_profit, stop_loss, score, "long", context)
                    engine.stats["entries_long"] += 1

                else:
                    engine.trade_manager.open_trade(ticker,  fill_price, size, take_profit, stop_loss, score, "short", context)
                    engine.stats["entries_short"] += 1