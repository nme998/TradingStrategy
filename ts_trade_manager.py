class Trade:
    next_id = 0
    def __init__(self, entry_price, size, direction, stop_loss, take_profit, entry_date, strategy, type, pair_id=None):
        self.trade_id = Trade.next_id
        Trade.next_id += 1
        self.entry_price = entry_price
        self.size = size
        self.direction = direction
        self.pair_id = pair_id
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
        self.max_risk = None
        self.linked_option_trade = None

        self.holding_days = 0

        self.prediction = None
        self.signal = None
        self.confidence = None
        self.regime = None
        self.entropy = None
        self.trend = None


class HedgePosition:
    next_id = 0

    def __init__(self, ticker, option_trade_id, quantity, entry_price, entry_date):
        self.hedge_id = HedgePosition.next_id
        HedgePosition.next_id += 1

        self.ticker = ticker
        self.pair_id = option_trade_id

        # Signed:
        # +50 = long 50 shares
        # -50 = short 50 shares
        self.quantity = quantity

        self.average_entry_price = entry_price
        self.current_price = entry_price

        self.entry_date = entry_date
        self.last_update_date = entry_date

        self.realized_pnl = 0.0
        self.transaction_cost = 0.0

        self.is_open = True

    @property
    def unrealized_pnl(self):
        return (
            self.current_price - self.average_entry_price
        ) * self.quantity

    @property
    def total_pnl(self):
        return self.realized_pnl + self.unrealized_pnl

class TradeManager:

    def __init__(self, engine):
        self.engine = engine

    def open_trade(self, ticker, fill_price, size, take_profit, stop_loss, score, type, context, pair_id=None):
        direction = 1 if context.signal > 0 else -1
        slippage = context.atr * self.engine.slippage_factor
        if size <= 0:
            return

        trade = Trade(fill_price, size, direction, stop_loss, take_profit, context.date, context.regime, type, pair_id)
        if ticker not in self.engine.open_trades:
            self.engine.open_trades[ticker] = []
        self.engine.open_trades[ticker].append(trade)

        trade.entry_slippage = slippage
        trade.prediction = context.prediction
        trade.signal = context.signal
        trade.confidence = context.confidence
        trade.regime = context.regime
        trade.entropy = context.entropy
        trade.trend = context.trend
        trade.score = score

        trade.max_risk = abs(trade.entry_price - trade.stop_loss) * trade.size
        self.engine.portfolio_risk += trade.max_risk

        side = "LONG" if direction == 1 else "SHORT"
        #print(f"{side} Signal: {ticker} |  Date: {context.date} | Price: {fill_price:.2f} | Size: {size:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Confidence: {context.confidence:.2f}")

    def close_trade(self, ticker, trade, price, date, reason):
        atr = self.engine.get_atr(ticker)
        slippage = atr * self.engine.slippage_factor

        if trade.direction == 1:
            exit_fill = price - slippage
        else:
            exit_fill = price + slippage

        trade_value = (abs(trade.entry_price * trade.size) + abs(exit_fill * trade.size))
        transaction_cost = (trade_value * self.engine.transaction_cost_pct)

        trade.is_open = False
        trade.exit_price = exit_fill
        trade.exit_date = date
        trade.exit_slippage = slippage
        trade.transaction_cost = transaction_cost

        gross_pnl = ((exit_fill - trade.entry_price) * trade.size * trade.direction)
        trade.pnl = gross_pnl - transaction_cost
        self.engine.capital += trade.pnl
        self.engine.portfolio_risk -= trade.max_risk

        if ticker not in self.engine.closed_trades:
            self.engine.closed_trades[ticker] = []

        self.engine.closed_trades[ticker].append(trade)

        #print(f"[{ticker}] [{reason}] EntryDate ({trade.entry_date}) Entry {trade.entry_price:.2f} → ExitDate ({trade.exit_date}) Exit {price:.2f} | PnL {trade.pnl:.2f}")

    def open_hedge(self, ticker, price, quantity, date, linked_option_trade):
        if abs(quantity) <= 0:
            return None

        slippage = price * self.engine.slippage_factor

        if quantity > 0:
            # Buying stock
            fill = price + slippage
        else:
            # Selling / shorting stock
            fill = price - slippage

        hedge = HedgePosition(ticker=ticker, option_trade_id=linked_option_trade.trade_id, quantity=quantity, entry_price=fill, entry_date=date)

        if ticker not in self.engine.open_hedges:
            self.engine.open_hedges[ticker] = []

        self.engine.open_hedges[ticker].append(hedge)

        # Stock has delta of 1 per share
        self.engine.portfolio_delta += quantity

        return hedge
    
    def close_hedge(self, ticker, hedge, price, date, reason="OPTION_CLOSE"):
        if not hedge.is_open:
            return

        quantity = hedge.quantity

        if quantity == 0:
            return

        slippage = price * self.engine.slippage_factor

        if quantity > 0:
            # Sell long shares
            fill = price - slippage

            pnl = (
                fill - hedge.average_entry_price
            ) * quantity

        else:
            # Buy back short shares
            fill = price + slippage

            pnl = (hedge.average_entry_price - fill) * abs(quantity)

        transaction_cost = (abs(quantity) * fill * self.engine.transaction_cost_pct)

        net_pnl = pnl - transaction_cost

        hedge.realized_pnl += net_pnl
        hedge.transaction_cost += transaction_cost

        hedge.current_price = price

        # Remove hedge delta from portfolio
        self.engine.portfolio_delta -= quantity

        # Add ONLY this final realised PnL
        self.engine.capital += net_pnl

        hedge.quantity = 0
        hedge.is_open = False
        hedge.last_update_date = date

        if ticker in self.engine.open_hedges:
            if hedge in self.engine.open_hedges[ticker]:
                self.engine.open_hedges[ticker].remove(hedge)

        if ticker not in self.engine.closed_hedges:
            self.engine.closed_hedges[ticker] = []

        self.engine.closed_hedges[ticker].append(hedge)

        print(
            f"[{ticker}] [{reason}] HEDGE "
            f"PnL {net_pnl:.2f} | "
            f"Total hedge PnL {hedge.realized_pnl:.2f}"
        )

    def rebalance_hedge(self, ticker, hedge, target_quantity, price, date):
        current_quantity = hedge.quantity

        if abs(target_quantity - current_quantity) < 1e-8:
            hedge.current_price = price
            hedge.last_update_date = date
            return

        adjustment = target_quantity - current_quantity

        slippage = price * self.engine.slippage_factor

        # ---------------------------------------------------------
        # Determine fill price
        # ---------------------------------------------------------

        if adjustment > 0:
            fill = price + slippage
        else:
            fill = price - slippage

        closing_quantity = 0.0
        opening_quantity = 0.0

        # Same direction -> entirely opening
        if current_quantity == 0:
            opening_quantity = abs(adjustment)

        elif current_quantity > 0 and adjustment > 0:
            # Long -> longer
            opening_quantity = adjustment

        elif current_quantity < 0 and adjustment < 0:
            # Short -> more short
            opening_quantity = abs(adjustment)

        else:
            # We're reducing or reversing the existing position
            closing_quantity = min(
                abs(current_quantity),
                abs(adjustment)
            )

            # If adjustment is larger than existing position,
            # the remainder opens the opposite side
            opening_quantity = max(
                abs(adjustment) - abs(current_quantity),
                0
            )

        # ---------------------------------------------------------
        # Realise PnL on the portion being closed
        # ---------------------------------------------------------

        realized_pnl = 0.0

        if closing_quantity > 0:
            if current_quantity > 0:
                # Closing a LONG
                realized_pnl = (fill - hedge.average_entry_price) * closing_quantity

            else:
                # Closing a SHORT
                realized_pnl = (hedge.average_entry_price - fill) * closing_quantity


        transaction_cost = (abs(adjustment) * fill * self.engine.transaction_cost_pct)
        net_realized_pnl = realized_pnl - transaction_cost

        hedge.realized_pnl += net_realized_pnl
        hedge.transaction_cost += transaction_cost

        self.engine.capital += net_realized_pnl

        # If we are adding to the same position
        if (current_quantity > 0 and target_quantity > current_quantity):
            old_value = (current_quantity * hedge.average_entry_price)
            new_value = (opening_quantity * fill)

            hedge.average_entry_price = (old_value + new_value) / target_quantity

        elif (current_quantity < 0 and target_quantity < current_quantity):
            old_abs_quantity = abs(current_quantity)

            old_value = (old_abs_quantity * hedge.average_entry_price)

            new_value = (opening_quantity * fill)

            hedge.average_entry_price = (old_value + new_value) / abs(target_quantity)

        # If we've completely closed and opened the opposite direction
        elif (current_quantity > 0 and target_quantity < 0):
            hedge.average_entry_price = fill

        elif (current_quantity < 0 and target_quantity > 0):
            hedge.average_entry_price = fill

        elif current_quantity == 0:
            hedge.average_entry_price = fill

        self.engine.portfolio_delta += (target_quantity - current_quantity)

        hedge.quantity = target_quantity
        hedge.current_price = price
        hedge.last_update_date = date