import numpy as np
from ts_strategies.ts_options.black_scholes import black_scholes_call, black_scholes_put
from ts_strategies.ts_options.greeks import calculate_delta, calculate_gamma, calculate_theta, calculate_vega
from ts_strategies.ts_options.options_data_loader import OptionDataLoader
from ts_trade_manager_options import OptionTradeManager

class OptionsTradeContext:

    def __init__(self, predicted_vol, implied_vol, option_price, bs_price,
                strike, option_type,  expiry, time_to_expiry, delta, gamma,
                theta, vega, confidence, entry_score, date):

            self.predicted_vol = predicted_vol
            self.implied_vol = implied_vol

            self.option_price = option_price
            self.bs_price = bs_price

            self.strike = strike
            self.option_type = option_type
            self.expiry = expiry
            self.time_to_expiry = time_to_expiry

            self.delta = delta
            self.gamma = gamma
            self.theta = theta
            self.vega = vega

            self.confidence = confidence
            self.entry_score = entry_score

            self.date = date

class OptionsVolatility:

    def __init__(self, iv_threshold=0.03, min_open_interest=100, min_volume=10):

        self.loader = OptionDataLoader()
        self.trade_manager = None

        self.max_entry_score = 7
        self.iv_threshold = iv_threshold
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume
        self.risk_free_rate = 0.03

        self.exit_threshold = 4

    def select_contract(self, option_chain, stock_price, option_type, mode="ATM"):
        chain = option_chain.copy()
        chain = chain[
            (chain["volume"] >= self.min_volume) &
            (chain["openInterest"] >= self.min_open_interest)
        ]

        if len(chain) == 0:
            return None

        chain = chain[chain["optionType"] == option_type]

        if len(chain) == 0:
            return None

        if mode == "ATM":
            chain["distance"] = abs(chain["strike"] - stock_price)
            return chain.sort_values("distance").iloc[0]

        if mode == "OTM":
            if option_type == "call":
                chain = chain[chain["strike"] > stock_price].sort_values("strike")
            else:
                chain = chain[chain["strike"] < stock_price].sort_values("strike", ascending=False)

            if len(chain) == 0:
                return None

            return chain.iloc[0]

        return None
    
    def compute_entry_score(self, predicted_vol, implied_vol, market_option_price, bs_price, delta, theta, vega, regime):
        score = 0

        vol_edge = predicted_vol - implied_vol

        if abs(vol_edge) > 0.03:
            score += 2

        if abs(vol_edge) > 0.05:
            score += 1


        mispricing = (market_option_price - bs_price) / bs_price

        if abs(mispricing) > 0.05:
            score += 2


        if regime == "high_vol":
            score += 1

        greek_quality = True

        # Avoid options with almost no exposure
        if abs(delta) < 0.20:
            greek_quality = False

        # Avoid contracts with almost no volatility sensitivity
        if abs(vega) < 0.05:
            greek_quality = False

        # Avoid excessive time decay
        if theta < -0.10:
            greek_quality = False

        if greek_quality:
            score += 1

        return score
    
    def compute_exit_score( self,  predicted_vol,  implied_vol,  market_option_price,  bs_price,  confidence,  theta):

        score = 0

        vol_edge = abs(predicted_vol - implied_vol)

        if vol_edge < 0.01:
            score += 2

        if bs_price <= 0:
            mispricing = 0
        else:
            mispricing = abs((market_option_price - bs_price) / bs_price)

        if mispricing < 0.02:
            score += 2

        if confidence < 0.40:
            score += 1

        if theta < -0.15:
            score += 1

        return score
    
    def calculate_confidence(self, vol_edge, mispricing, delta, theta, vega, regime):

        vol_score = min(abs(vol_edge) / 0.10, 1.0)
        mispricing_score = min(abs(mispricing) / 0.10, 1.0)

        if regime == "high_vol":
            regime_score = 1.0
        else:
            regime_score = 0

        greek_score = 0.0

        if abs(delta) >= 0.20:
            greek_score += 0.40

        if abs(vega) >= 0.05:
            greek_score += 0.30

        if theta >= -0.10:
            greek_score += 0.30

        confidence = (0.40 * vol_score + 0.30 * mispricing_score + 0.20 * regime_score + 0.10 * greek_score)

        return np.clip(confidence, 0.0, 1.0)
    
    def calculate_position_size(self, engine, option_price, confidence, entry_score):

        risk_amount = (
            engine.capital *
            engine.risk_per_trade
        )
        #print(f"Base risk amount: {risk_amount:.2f} for capital {engine.capital:.2f}")
        risk_amount *= confidence
        risk_amount *= engine.score_to_multiplier(entry_score)

        if option_price <= 0:
            return 0

        contracts = risk_amount / (option_price * 100)
        max_contracts = int(engine.capital / (option_price * 100))
        contracts = min(contracts, max_contracts)

        #print(f"Calculated position size: {contracts} contracts for option price {option_price:.2f}, confidence {confidence:.2f}, entry score {entry_score}, risk amount {risk_amount:.2f}, max contracts {max_contracts}")

        return min(50, int(contracts))
    
    def process_entries(self, engine, rows, volatility_predictions, return_predictions, regime, lookback_windows):
        
        for ticker, row in rows.items():
            if ticker not in volatility_predictions:
                continue

            entry_score = 0
            direction = (0.5 * return_predictions[ticker][0] 
                         +  0.3 * return_predictions[ticker][1] 
                         +  0.2 * return_predictions[ticker][2])
            if direction >= 0:
                option_type = "call"
            else:
                option_type = "put"

            stock_price = row["Close"]

            prediction = volatility_predictions[ticker]
            predicted_vol = ( 0.5 * prediction[0] + 0.3 * prediction[1] + 0.2 * prediction[2])

            contract = self.loader.generate_contract(stock_price=stock_price, current_date=row.name, 
                                                     option_type=option_type, mode="ATM")

            if contract is None:
                continue

            while True:
                strike = contract["strike"]
                expiry = contract["expiry"]
                option_type = contract["optionType"]
                time_to_expiry = contract["time_to_expiry"]
                historical_vol = lookback_windows[ticker].iloc[-1]["rv20"]
                vrp = 0.15 * historical_vol
                noise = np.random.normal(0, 0.02)
                implied_vol = historical_vol + vrp + noise
                implied_vol = max(implied_vol, 0.05)

                if option_type == "call":
                    option_price = black_scholes_call(stock_price, strike, time_to_expiry, self.risk_free_rate, implied_vol)
                    bs_price = black_scholes_call(stock_price, strike, time_to_expiry, self.risk_free_rate, predicted_vol)
                else:
                    option_price = black_scholes_put(stock_price, strike, time_to_expiry, self.risk_free_rate, implied_vol)
                    bs_price = black_scholes_put(stock_price, strike, time_to_expiry, self.risk_free_rate, predicted_vol)

                if bs_price <= 0 or option_price <= 0:
                    break


                if option_price < 0.05:
                    break

                delta = calculate_delta(stock_price, strike, time_to_expiry, self.risk_free_rate, predicted_vol, option_type)
                gamma = calculate_gamma(stock_price, strike, time_to_expiry, self.risk_free_rate, predicted_vol)
                theta = calculate_theta(stock_price, strike, time_to_expiry, self.risk_free_rate, predicted_vol, option_type)
                vega = calculate_vega(stock_price, strike, time_to_expiry, self.risk_free_rate, predicted_vol)
                entry_score = self.compute_entry_score(predicted_vol, implied_vol, option_price, 
                                                 bs_price, delta, theta, vega, regime)

                if entry_score < engine.entry_threshold:
                    break

                confidence = self.calculate_confidence(
                    predicted_vol - implied_vol,
                    (option_price - bs_price) / bs_price,
                    delta,
                    theta,
                    vega,
                    regime
                )

                if (confidence >= 0.99 and entry_score == self.max_entry_score and contract.get("selection") != "OTM"):
                    new_contract = self.loader.generate_contract(stock_price=stock_price, current_date=row.name, 
                                                     option_type=option_type, mode="OTM")

                    if new_contract is not None:
                        new_contract["selection"] = "OTM"
                        contract = new_contract
                        continue

                break

            if entry_score < engine.entry_threshold:
                continue

            size = self.calculate_position_size(engine, option_price, confidence, entry_score)
            #print(f"Ticker: {ticker}, Option Type: {option_type}, Strike: {strike:.2f}, Expiry: {expiry}, Option Price: {option_price:.2f}, BS Price: {bs_price:.2f}, Predicted Vol: {predicted_vol:.4f}, Implied Vol: {implied_vol:.4f}, Entry Score: {entry_score}, Confidence: {confidence:.4f}, Position Size: {size}")

            if size <= 0 or confidence < 0.4 or option_price < 0.05:
                continue

            context = OptionsTradeContext(predicted_vol=predicted_vol, implied_vol=implied_vol,
                option_price=option_price, bs_price=bs_price, strike=strike, option_type=option_type,
                expiry=expiry, time_to_expiry=time_to_expiry, delta=delta, gamma=gamma, theta=theta,
                vega=vega, confidence=confidence, entry_score=entry_score, date=row.name)

            self.trade_manager.open_option_trade(ticker=ticker, underlying_price=row["Close"], 
                                                 contract=contract, size=size, context=context)

    def update_trades(self, engine, ticker, row, volatility_prediction, regime, lookback_windows):

        if ticker not in engine.open_options_trades:
            return

        for trade in engine.open_options_trades[ticker][:]:
            stock_price = row["Close"]
            predicted_vol = (
                0.5 * volatility_prediction[0] +
                0.3 * volatility_prediction[1] +
                0.2 * volatility_prediction[2]
            )
            trade.time_to_expiry -= 1 / 252
            trade.time_to_expiry = max(trade.time_to_expiry, 0)

            contract = self.loader.generate_contract(stock_price=stock_price, current_date=row.name, 
                                                     option_type=trade.option_type, mode="ATM")

            if len(contract) == 0:
                continue

            #contract = contract.iloc[0]

            historical_vol = lookback_windows[ticker].iloc[-1]["rv20"]
            vrp = 0.15 * historical_vol
            noise = np.random.normal(0, 0.02)
            implied_vol = historical_vol + vrp + noise
            implied_vol = max(implied_vol, 0.05)

            if trade.option_type == "call":
                option_price = black_scholes_call(stock_price, trade.strike, trade.time_to_expiry, self.risk_free_rate, implied_vol)
            else:
                option_price = black_scholes_put(stock_price, trade.strike, trade.time_to_expiry, self.risk_free_rate, implied_vol)

            if option_price <= 0:
                continue

            delta = calculate_delta(stock_price, trade.strike, trade.time_to_expiry, 
                                    self.risk_free_rate, predicted_vol, trade.option_type)

            theta = calculate_theta(stock_price, trade.strike, trade.time_to_expiry,
                                    self.risk_free_rate, predicted_vol, trade.option_type)

            vega = calculate_vega(stock_price, trade.strike, trade.time_to_expiry,
                                    self.risk_free_rate, predicted_vol)

            confidence = self.calculate_confidence(
                predicted_vol - implied_vol,
                (option_price - trade.entry_price) / trade.entry_price,
                delta,
                theta,
                vega,
                regime
            )

            unrealized_return = (option_price - trade.entry_price) / trade.entry_price

            exit_score = self.compute_exit_score(predicted_vol, implied_vol, option_price,
                                                    trade.entry_price, confidence, theta)
            theta_ratio = abs(theta) / option_price

            if exit_score >= self.exit_threshold:
                trade.exit_implied_vol = implied_vol
                trade.exit_predicted_vol = predicted_vol
                self.trade_manager.close_option_trade(ticker, trade, option_price, row["Close"], row.name, "EXIT_SCORE")
                continue

            if unrealized_return >= 0.4:
                trade.exit_implied_vol = implied_vol
                trade.exit_predicted_vol = predicted_vol
                self.trade_manager.close_option_trade(ticker, trade, option_price, row["Close"], row.name, "PROFIT_TARGET")
                continue

            if theta_ratio > 0.04:
                trade.exit_implied_vol = implied_vol
                trade.exit_predicted_vol = predicted_vol
                self.trade_manager.close_option_trade(ticker, trade, option_price, row["Close"], row.name, "THETA_EXIT")
                continue

            if trade.time_to_expiry <= (5 / 252):
                trade.exit_implied_vol = implied_vol
                trade.exit_predicted_vol = predicted_vol
                self.trade_manager.close_option_trade(ticker, trade, option_price, row["Close"], row.name, "EXPIRY")
                continue

            trade.current_price = option_price