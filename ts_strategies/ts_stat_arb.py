from ts_stock_relation import CointegrationEngine, OUModel
import pandas as pd


class StatArb:
    def __init__(self, tickers):
        self.tickers = tickers
        self.ou_model = OUModel()

        self.last_rebalance = None
        self.pairs = None
        self.active_pairs = set()
        self.relation_engine = None

    def refresh_pairs(self, current_date, stat_arb_lookbacks=None):
        if (self.last_rebalance is None or (current_date - self.last_rebalance).days >= 90):
            prices = {}
            for ticker, df in stat_arb_lookbacks.items():
                prices[ticker] = (df.set_index(df.index)["Close"])
            prices = pd.DataFrame(prices)

            self.relation_engine = CointegrationEngine(tickers=self.tickers, date=current_date, prices=prices)
            self.pairs = self.relation_engine.find_cointegrated_pairs()
            self.last_rebalance = current_date
            if self.pairs is None or self.pairs.empty:
                print("No pairs found")
            else:
                for _, pair in self.pairs.iterrows():
                    print(
                        f"{pair['asset_1']:5} | "
                        f"{pair['asset_2']:5} | "
                        f"Coint p={pair['coint_pvalue']:.4f} | "
                        f"HalfLife={pair['half_life']:.1f}"
                    )

    def calculate_pair_size(self, engine, context, ticker, score=8):
        price = engine.current_prices[ticker]
        risk_amount = engine.capital * engine.risk_per_trade
        vol = engine.get_volatility(ticker)

        if vol is None or vol <= 0:
            return 0

        target_vol = 0.01
        vol_adjustment = target_vol / (vol + 1e-8)
        score_multiplier = engine.score_to_multiplier(score)
        base_size = risk_amount / (price * vol + 1e-8)
        size = base_size * score_multiplier * vol_adjustment

        if context.entropy is not None:
            if context.entropy < 0.90:
                size *= 1.5
            elif context.entropy < 0.93:
                size *= 1.7
            elif context.entropy < 0.95:
                size *= 0.5
            else:
                size *= 0.8

        return size
    
    def calcuate_limit_prices(self, engine, ticker, rows, price, type, context):
        price = rows[ticker]["Close"]
        slippage = context.atr * engine.slippage_factor
        if type == "long":
            fill_price = price + slippage
            stop_loss = None #fill_price - 1.5 * context.atr
            take_profit = None #fill_price + 2.5 * context.atr
        elif type == "short":
            fill_price = price - slippage
            stop_loss = None #fill_price + 1.5 * context.atr
            take_profit = None #fill_price - 2.5 * context.atr

        return fill_price,stop_loss, take_profit
    
    def update_trades(self, engine, ticker_contexts, current_date, stat_arb_lookbacks=None):
        if len(self.active_pairs) == 0:
            return

        for _, pair in self.pairs.iterrows():

            asset_1 = pair["asset_1"]
            asset_2 = pair["asset_2"]
            beta = pair["beta"]
            intercept = pair["intercept"]

            pair_id = tuple(sorted([asset_1, asset_2]))

            if asset_1 not in ticker_contexts or asset_2 not in ticker_contexts:
                continue

            spread = self.ou_model.build_spread(ticker_contexts[asset_1].lookback_window, ticker_contexts[asset_2].lookback_window, beta, intercept)
            current_spread = spread.iloc[-1]
            z = (current_spread - pair["ou_mu"]) / pair["ou_sigma"]

            if abs(z) < 0.1:
                print(f"Closing pair {asset_1} & {asset_2} | Z-Score: {z:.2f}")
                if asset_1 in engine.open_trades:
                    print(f"Closing trades for {asset_1}")
                    for trade in engine.open_trades[asset_1][:]:
                        if trade.pair_id != pair_id:
                            print(f"Skipping trade for {asset_1} with pair_id {trade.pair_id} (looking for {pair_id})")
                            continue
                        engine.close_trade(asset_1, trade, engine.current_prices[asset_1], current_date, "MEAN_REVERSION")
                        engine.open_trades[asset_1].remove(trade)
                        print(f"Closed trade for {asset_1} with pair_id {trade.pair_id}")

                if asset_2 in engine.open_trades:
                    for trade in engine.open_trades[asset_2][:]:
                        if trade.pair_id != pair_id:
                            continue
                        engine.close_trade(asset_2, trade, engine.current_prices[asset_2], current_date, "MEAN_REVERSION")
                        engine.open_trades[asset_2].remove(trade)
                        print(f"Closed trade for {asset_2} with pair_id {trade.pair_id}")

                self.active_pairs.remove(pair_id)

    def process_entries(self, engine, rows, ticker_contexts, current_date, stat_arb_lookbacks=None):
        self.refresh_pairs(current_date, stat_arb_lookbacks)
        if self.pairs is None or self.pairs.empty:
            return

        for _, pair in self.pairs.iterrows():
            asset_1 = pair["asset_1"]
            asset_2 = pair["asset_2"]
            beta = pair["beta"]
            intercept = pair["intercept"]

            pair_id = tuple(sorted([asset_1, asset_2]))
            if pair_id in self.active_pairs:
                print(f"Skipping pair {asset_1} & {asset_2} | Already active")
                continue

            if asset_1 not in ticker_contexts or asset_2 not in ticker_contexts:
                print(f"Skipping pair {asset_1} & {asset_2} | Missing context")
                continue

            spread = self.ou_model.build_spread(ticker_contexts[asset_1].lookback_window, ticker_contexts[asset_2].lookback_window, beta, intercept)
            current_spread = spread.iloc[-1]
            z = (current_spread - pair["ou_mu"]) / pair["ou_sigma"]
            print(f"Checking pair {asset_1} & {asset_2} | Z-Score: {z:.2f} +++++++++++++++++")

            ctx1 = ticker_contexts[asset_1]
            ctx2 = ticker_contexts[asset_2]

            size1 = self.calculate_pair_size(engine, ctx1, asset_1)
            price1 = engine.current_prices[asset_1]
            price2 = engine.current_prices[asset_2]
            size2 = size1 * abs(beta)

            #print(asset_1, asset_2, beta, z)

            if z > 1:
                asset_1_fill_price, asset_1_stop_loss, asset_1_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_1, rows, engine.current_prices[asset_1], 
                                                                                    "short", ctx1
                                                                                    )
                asset_2_fill_price, asset_2_stop_loss, asset_2_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_2, rows, engine.current_prices[asset_2], 
                                                                                    "long", ctx2
                                                                                    )

                engine.open_trade(asset_1, asset_1_fill_price, size1, asset_1_take_profit, asset_1_stop_loss, z, "short", ctx1, pair_id)
                engine.open_trade(asset_2, asset_2_fill_price, size2, asset_2_take_profit, asset_2_stop_loss, z, "long", ctx2, pair_id)
                self.active_pairs.add(pair_id)
                print(f"Opened pair {asset_1} (short) & {asset_2} (long) | Z-Score: {z:.2f}")
                print("AFTER ADD:", self.active_pairs)

            elif z < -1:
                asset_1_fill_price, asset_1_stop_loss, asset_1_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_1, rows, engine.current_prices[asset_1], 
                                                                                    "long", ctx1
                                                                                    )
                asset_2_fill_price, asset_2_stop_loss, asset_2_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_2, rows, engine.current_prices[asset_2], 
                                                                                    "short", ctx2
                                                                                    )

                engine.open_trade(asset_1, asset_1_fill_price, size1, asset_1_take_profit, asset_1_stop_loss, z, "long", ctx1, pair_id)
                engine.open_trade(asset_2, asset_2_fill_price, size2, asset_2_take_profit, asset_2_stop_loss, z, "short", ctx2, pair_id)
                self.active_pairs.add(pair_id)
                print(f"Opened pair {asset_1} (long) & {asset_2} (short) | Z-Score: {z:.2f}")
                print("AFTER ADD:", self.active_pairs)