import pandas as pd


class OptionDataLoader:

    def __init__(self,
                 expiry_days=30,
                 strike_spacing=5,
                 otm_steps=3):

        self.expiry_days = expiry_days
        self.strike_spacing = strike_spacing
        self.otm_steps = otm_steps


    def generate_contract(self,
                          stock_price,
                          current_date,
                          option_type,
                          mode="ATM"):

        atm = round(stock_price / self.strike_spacing) * self.strike_spacing

        if mode == "ATM":
            strike = atm

        elif mode == "OTM":

            if option_type == "call":
                strike = atm + self.otm_steps * self.strike_spacing

            else:
                strike = atm - self.otm_steps * self.strike_spacing

        else:
            raise ValueError("mode must be ATM or OTM")

        expiry = current_date + pd.Timedelta(days=self.expiry_days)

        return {
            "strike": strike,
            "expiry": expiry,
            "optionType": option_type,
            "time_to_expiry": self.expiry_days / 365.0
        }