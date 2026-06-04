from ts_stock_relation import CointegrationEngine, OUModel
import pandas as pd


class StatArb:
    def __init__(self, tickers, date):
        self.tickers = tickers
        self.ou_model = OUModel()

        self.last_rebalance = None
        self.pairs = None
        self.relation_engine = None

    def refresh_pairs(self, current_date):
        if (self.last_rebalance is None or (current_date - self.last_rebalance).days >= 90):
            self.relation_engine = CointegrationEngine(tickers=self.tickers, date=current_date)
            self.pairs = self.relation_engine.find_cointegrated_pairs()
            self.last_rebalance = current_date

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
            stop_loss = fill_price - 1.5 * context.atr
            take_profit = fill_price + 2.5 * context.atr
        elif type == "short":
            fill_price = price - slippage
            stop_loss = fill_price + 1.5 * context.atr
            take_profit = fill_price - 2.5 * context.atr

        return fill_price,stop_loss, take_profit

    def process_entries(self, engine, rows, ticker_contexts, current_date):
        self.refresh_pairs(current_date)
        if self.pairs is None or self.pairs.empty:
            return
        

        for _, pair in self.pairs.iterrows():
            asset_1 = pair["asset_1"]
            asset_2 = pair["asset_2"]
            beta = pair["beta"]

            if asset_1 not in ticker_contexts or asset_2 not in ticker_contexts:
                continue

            spread = self.relation_engine.build_spread(asset_1, asset_2, beta)
            params = self.ou_model.fit(spread)
            z = self.ou_model.ou_zscore(spread, params)

            ctx1 = ticker_contexts[asset_1]
            ctx2 = ticker_contexts[asset_2]

            size1 = self.calculate_pair_size(engine, ctx1, asset_1)
            size2 = self.calculate_pair_size(engine, ctx2, asset_2)

            if z > 2:
                asset_1_fill_price, asset_1_stop_loss, asset_1_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_1, rows, engine.current_prices[asset_1], 
                                                                                    "short", ctx1
                                                                                    )
                asset_2_fill_price, asset_2_stop_loss, asset_2_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_2, rows, engine.current_prices[asset_2], 
                                                                                    "long", ctx2
                                                                                    )

                engine.open_trade(asset_1, asset_1_fill_price, size1, asset_1_take_profit, asset_1_stop_loss, z, "short", ctx1)
                engine.open_trade(asset_2, asset_2_fill_price, size2, asset_2_take_profit, asset_2_stop_loss, z, "long", ctx2)

            elif z < -2:
                asset_1_fill_price, asset_1_stop_loss, asset_1_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_1, rows, engine.current_prices[asset_1], 
                                                                                    "long", ctx1
                                                                                    )
                asset_2_fill_price, asset_2_stop_loss, asset_2_take_profit = self.calcuate_limit_prices(
                                                                                    engine, asset_2, rows, engine.current_prices[asset_2], 
                                                                                    "short", ctx2
                                                                                    )

                engine.open_trade(asset_1, asset_1_fill_price, size1, asset_1_take_profit, asset_1_stop_loss, z, "long", ctx1)
                engine.open_trade(asset_2, asset_2_fill_price, size2, asset_2_take_profit, asset_2_stop_loss, z, "short", ctx2)