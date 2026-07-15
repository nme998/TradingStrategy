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

        self.holding_days = 0

        self.prediction = None
        self.signal = None
        self.confidence = None
        self.regime = None
        self.entropy = None
        self.trend = None

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

        if ticker not in self.engine.closed_trades:
            self.engine.closed_trades[ticker] = []

        self.engine.closed_trades[ticker].append(trade)

        print(f"[{ticker}] [{reason}] EntryDate ({trade.entry_date}) Entry {trade.entry_price:.2f} → ExitDate ({trade.exit_date}) Exit {price:.2f} | PnL {trade.pnl:.2f}")
   