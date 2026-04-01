import requests
import pandas as pd
from datetime import datetime

API_KEY = "2QI86EFCIC70I16M"
symbol = "AAPL"

url = "https://www.alphavantage.co/query"

news_params = {
    "function": "NEWS_SENTIMENT",
    "tickers": symbol,
    "time_from": "20210308T0130",
    "time_to": "20260304T0130",
    "apikey": API_KEY
}

company_params = {
    "function": "OVERVIEW",
    "symbol": symbol,
    "apikey": API_KEY
}

response = requests.get(url, params=news_params)
news_data = response.json()
response = requests.get(url, params=company_params)
company_data = response.json()

news_items = news_data.get("feed", [])

filtered_data = []

for item in news_items:
    time_published = item.get("time_published")

    confidence = None
    for ticker_info in item.get("ticker_sentiment", []):
        if ticker_info.get("ticker") == "AAPL":
            confidence = ticker_info.get("relevance_score")
            sentiment_score = ticker_info.get("ticker_sentiment_score")
            break

    filtered_data.append({
        "time_published": time_published,
        "confidence": confidence,
        "sentiment_score": sentiment_score
    })

filtered_company_data = []
for item in company_data:
    sector = company_data.get("Sector"),
    book_value = company_data.get("BookValue"),
    PERatio = company_data.get("PERatio"),
    PEGRatio = company_data.get("PEGRatio"),
    analyst_target_price = company_data.get("AnalystTargetPrice"),
    profit_margin = company_data.get("ProfitMargin"),
    operating_margin = company_data.get("OperatingMarginTTM"),
    return_on_assets = company_data.get("ReturnOnAssetsTTM"),
    return_on_equity = company_data.get("ReturnOnEquityTTM"),
    revenue = company_data.get("RevenueTTM"),
    gross_profit = company_data.get("GrossProfitTTM"),
    EVToRevenue = company_data.get("EVToRevenue")

    filtered_company_data.append({
        "sector": sector,
        "book_value": book_value,
        "PERatio": PERatio,
        "PEGRatio": PEGRatio,
        "analyst_target_price": analyst_target_price,
        "profit_margin": profit_margin,
        "operating_margin": operating_margin,
        "return_on_assets": return_on_assets,
        "return_on_equity": return_on_equity,
        "revenue": revenue,
        "gross_profit": gross_profit,
        "EVToRevenue": EVToRevenue
    })

print(filtered_data)
pd.set_option('display.max_colwidth', None) 
news_df = pd.DataFrame(filtered_data)
news_df = news_df.sort_values(by='time_published', ascending=False)


pd.set_option('display.max_colwidth', None)

print(news_df.head())

news_df['time_published'] = pd.to_datetime(news_df['time_published'], format="%Y%m%dT%H%M%S")

# Extract only the date (YYYY-MM-DD)
news_df['time_published'] = news_df['time_published'].dt.date
news_df.set_index('time_published', inplace=True)
print(news_df.head())

numeric_cols = news_df.select_dtypes(include='object').columns 
for col in numeric_cols:
    news_df[col] = pd.to_numeric(news_df[col], errors='coerce')
df_avg = news_df.groupby(news_df.index).mean()

print(df_avg.head())


#_____________________________________________________________________________________________

'''
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# -------------------------------
# 1. PREPARE DATA
# -------------------------------

end_date = datetime.datetime.now() 
start_date = end_date - datetime.timedelta(days = 365 * 5)
df = yf.download('AAPL', start=start_date.date(), end=end_date.date(), auto_adjust=False)
df.columns = df.columns.get_level_values(0)

# Compute log returns
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

returns = df['log_return'].values.reshape(-1, 1)

# -------------------------------
# 2. FIT 3-STATE GAUSSIAN HMM
# -------------------------------

hmm_model = GaussianHMM(
    n_components=2,        # 3 hidden states
    covariance_type="full",
    n_iter=1000,
    random_state=42
)

hmm_model.fit(returns)

# -------------------------------
# 3. EXTRACT MODEL PARAMETERS
# -------------------------------

transition_matrix = hmm_model.transmat_
state_means = hmm_model.means_.flatten()
state_vars = np.array([np.diag(cov)[0] for cov in hmm_model.covars_])

print("Transition Matrix:")
print(transition_matrix)

print("\nState Means:")
print(state_means)

print("\nState Variances:")
print(state_vars)

# -------------------------------
# 4. PREDICT STATES & PROBABILITIES
# -------------------------------

hidden_states = hmm_model.predict(returns)
state_probs = hmm_model.predict_proba(returns)

df['regime'] = hidden_states
df['prob_state_0'] = state_probs[:, 0]
df['prob_state_1'] = state_probs[:, 1]
#df['prob_state_2'] = state_probs[:, 2]

# -------------------------------
# 5. MAP STATES TO UP / FLAT / DOWN
# -------------------------------

sorted_states = np.argsort(state_means)

down_state = sorted_states[0]
#flat_state = sorted_states[1]
up_state = sorted_states[1]

print("\nState Mapping:")
print("Down state:", down_state)
#print("Flat state:", flat_state)
print("Up state:", up_state)

# -------------------------------
# 6. VALIDATION CHECKS
# -------------------------------

print("\nValidation Statistics:")
for i in range(2):
    state_returns = df[df['regime'] == i]['log_return']
    print(f"State {i}:")
    print("  Mean:", state_returns.mean())
    print("  Std:", state_returns.std())
    print("  Count:", len(state_returns))

print("\nDiagonal (Persistence) Probabilities:")
print(np.diag(transition_matrix))

# -------------------------------
# 7. OPTIONAL: VISUALIZE REGIMES
# -------------------------------

plt.figure(figsize=(12,6))
plt.plot(df['Close'], label="Close Price")

for i in range(2):
    plt.scatter(
        df.index[df['regime'] == i],
        df['Close'][df['regime'] == i],
        s=5
    )

plt.legend()
plt.title("Price with Hidden Regimes")
plt.show()
'''