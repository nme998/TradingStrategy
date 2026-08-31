class OptionTrade:
    next_id = 0

    def __init__(self, entry_price, underlying_price, size, direction, entry_date, option_type):

        self.trade_id = OptionTrade.next_id
        OptionTrade.next_id += 1

        self.option_type = option_type
        self.entry_price = entry_price
        self.premium_paid = entry_price * size * 100
        self.exit_price = None
        self.premium_received = None
        self.size = size
        self.direction = direction

        self.pnl = 0
        self.entry_score = None
        self.time_to_expiry = None

        self.entry_date = entry_date
        self.exit_date = None

        self.is_open = True
        self.strike = None
        self.expiry = None

        self.underlying_entry = underlying_price
        self.underlying_exit = None

        self.implied_vol = None
        self.predicted_vol = None
        self.exit_implied_vol = None
        self.exit_predicted_vol = None

        self.delta = None
        self.gamma = None
        self.theta = None
        self.vega = None
        self.rho = None
        self.max_risk = None

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

class OptionTradeManager:

    def __init__(self, engine):
        self.engine = engine


    def open_option_trade(self, ticker, underlying_price, contract, size, context):

        if size <= 0:
            return

        trade = OptionTrade(entry_price=context.option_price, underlying_price=underlying_price, size=size, 
                            direction=1 if contract["optionType"] == "call" else -1, 
                            entry_date=context.date, option_type=contract["optionType"])

        trade.strike = context.strike
        trade.expiry = context.expiry
        trade.time_to_expiry = context.time_to_expiry

        trade.predicted_vol = context.predicted_vol
        trade.implied_vol = context.implied_vol

        trade.delta = context.delta
        trade.gamma = context.gamma
        trade.theta = context.theta
        trade.vega = context.vega

        trade.confidence = context.confidence
        trade.entry_score = context.entry_score

        premium_cost = context.option_price * size * 100
        self.engine.capital -= premium_cost

        multiplier = trade.size * 100
        trade.max_risk = premium_cost

        self.engine.portfolio_delta += trade.delta * multiplier
        self.engine.portfolio_gamma += trade.gamma * multiplier
        self.engine.portfolio_theta += trade.theta * multiplier
        self.engine.portfolio_vega += trade.vega * multiplier
        self.engine.portfolio_risk += trade.max_risk
        self.engine.portfolio_option_value -= trade.entry_price * multiplier

        if ticker not in self.engine.open_options_trades:
            self.engine.open_options_trades[ticker] = []

        self.engine.open_options_trades[ticker].append(trade)

        return trade

    def update_option_greeks(self, trade, delta, gamma, theta, vega):
        multiplier = trade.size * 100

        self.engine.portfolio_delta += (delta - trade.delta) * multiplier
        self.engine.portfolio_gamma += (gamma - trade.gamma) * multiplier
        self.engine.portfolio_theta += (theta - trade.theta) * multiplier
        self.engine.portfolio_vega += (vega - trade.vega) * multiplier

        trade.delta = delta
        trade.gamma = gamma
        trade.theta = theta
        trade.vega = vega


    def close_option_trade(self, ticker, trade, option_price, underlying_price, date, reason):
        trade.is_open = False

        option_slippage = option_price * self.engine.slippage_factor
        if trade.direction == 1:
            exit_fill = option_price - option_slippage
        else:
            exit_fill = option_price + option_slippage

        trade.exit_price = exit_fill
        trade.exit_date = date
        trade.premium_received = exit_fill * trade.size *100
        trade.underlying_exit = underlying_price
        trade.pnl = (exit_fill - trade.entry_price) * trade.size * 100

        self.engine.capital += trade.premium_received

        multiplier = trade.size * 100
        trade.max_risk = trade.premium_received

        self.engine.portfolio_delta -= trade.delta * multiplier
        self.engine.portfolio_gamma -= trade.gamma * multiplier
        self.engine.portfolio_theta -= trade.theta * multiplier
        self.engine.portfolio_vega -= trade.vega * multiplier
        self.engine.portfolio_risk -= trade.premium_received
        self.engine.portfolio_option_value -= trade.entry_price * multiplier

        if ticker not in self.engine.closed_options_trades:
            self.engine.closed_options_trades[ticker] = []

        self.engine.closed_options_trades[ticker].append(trade)

        # Remove from open trades
        if ticker in self.engine.open_options_trades:
            if trade in self.engine.open_options_trades[ticker]:
                self.engine.open_options_trades[ticker].remove(trade)



        print(
            ticker,
            "size:", trade.size,
            "premium_paid:", trade.premium_paid,
            "premium_received:", trade.premium_received,
            "current_capital:", self.engine.capital,
            "entry:", trade.entry_price,
            "exit:", exit_fill,
            "pnl:", trade.pnl
        )
        '''
        print(
            f"[{ticker}] [{reason}] "
            f"{trade.option_type.upper()} "
            f"Strike {trade.strike:.2f} | "
            f"Position Size {trade.size} | "
            f"EntryDate- {trade.entry_date.date()} "
            f"Entry- {trade.entry_price:.2f} → "
            f"ExitDate- {trade.exit_date.date()} "
            f"Exit- {trade.exit_price:.2f} | "
            f"PnL {trade.pnl:.2f}"
        )
        print("Current Capital:", self.engine.capital, " | Open Trades:", len(self.engine.open_options_trades.get(ticker, [])), 
              " | Closed Trades:", len(self.engine.closed_options_trades.get(ticker, [])))
        '''

    