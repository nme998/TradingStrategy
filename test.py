import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

API_KEY = "2QI86EFCIC70I16M"
BASE_URL = "https://www.alphavantage.co/query"

# =========================================================
# CONFIG
# =========================================================

SYMBOL = "AAPL"

# =========================================================
# HELPER
# =========================================================

def fetch_alpha_vantage(function, symbol):

    time.sleep(12)

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()
    print(function, data.keys())

    if (
        "Information" in data or
        "Note" in data or
        "Error Message" in data
    ):
        print("API RESPONSE ISSUE:")
        print(data)
        return {}

    return data

# =========================================================
# EARNINGS FEATURES
# =========================================================

def get_earnings_features(symbol):
    data = fetch_alpha_vantage("EARNINGS", symbol)

    quarterly = pd.DataFrame(data["quarterlyEarnings"])

    quarterly["reportedDate"] = pd.to_datetime(
        quarterly["reportedDate"]
    )

    numeric_cols = [
        "reportedEPS",
        "estimatedEPS",
        "surprise",
        "surprisePercentage"
    ]

    for col in numeric_cols:
        quarterly[col] = pd.to_numeric(
            quarterly[col],
            errors="coerce"
        )

    quarterly = quarterly.rename(columns={
        "reportedDate": "date"
    })

    quarterly["earnings_flag"] = 1

    return quarterly[[
        "date",
        "reportedEPS",
        "estimatedEPS",
        "surprise",
        "surprisePercentage",
        "earnings_flag"
    ]]

# =========================================================
# INCOME STATEMENT FEATURES
# =========================================================

def get_income_features(symbol):
    data = fetch_alpha_vantage("INCOME_STATEMENT", symbol)

    quarterly = pd.DataFrame(data["quarterlyReports"])

    quarterly["fiscalDateEnding"] = pd.to_datetime(
        quarterly["fiscalDateEnding"]
    )

    cols = [
        "totalRevenue",
        "grossProfit",
        "netIncome",
        "ebitda"
    ]

    for col in cols:
        quarterly[col] = pd.to_numeric(
            quarterly[col],
            errors="coerce"
        )

    quarterly = quarterly.rename(columns={
        "fiscalDateEnding": "date"
    })

    # Revenue growth
    quarterly["revenue_growth"] = (
        quarterly["totalRevenue"].pct_change()
    )

    # Net margin
    quarterly["net_margin"] = (
        quarterly["netIncome"] /
        quarterly["totalRevenue"]
    )

    # EBITDA growth
    quarterly["ebitda_growth"] = (
        quarterly["ebitda"].pct_change()
    )

    return quarterly[[
        "date",
        "revenue_growth",
        "net_margin",
        "ebitda_growth"
    ]]

# =========================================================
# OVERVIEW FEATURES
# =========================================================

def get_overview_features(symbol):
    data = fetch_alpha_vantage("OVERVIEW", symbol)

    overview = {
        "market_cap": pd.to_numeric(
            data.get("MarketCapitalization"),
            errors="coerce"
        ),
        "pe_ratio": pd.to_numeric(
            data.get("PERatio"),
            errors="coerce"
        ),
        "price_to_book": pd.to_numeric(
            data.get("PriceToBookRatio"),
            errors="coerce"
        ),
        "beta": pd.to_numeric(
            data.get("Beta"),
            errors="coerce"
        )
    }

    return overview

# =========================================================
# MERGE EVERYTHING
# =========================================================

def build_feature_set(symbol):

    # =====================================================
    # 1. LOAD FUNDAMENTALS (EVENT-BASED TABLES)
    # =====================================================

    earnings = get_earnings_features(symbol)
    income   = get_income_features(symbol)

    dfs = [earnings, income]

    clean_dfs = []

    # =====================================================
    # 2. CLEAN + STANDARDISE EACH DATAFRAME
    # =====================================================

    for df in dfs:
        if df is None or df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # keep ONLY real event rows (drop duplicates if same date appears)
        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

        clean_dfs.append(df)

    # =====================================================
    # 3. MERGE ALL EVENT DATA (NO DAILY GRID)
    # =====================================================

    if len(clean_dfs) == 0:
        return pd.DataFrame()

    fundamentals = pd.concat(clean_dfs, axis=0)
    fundamentals = fundamentals.sort_values("date")
    fundamentals = fundamentals.drop_duplicates(subset=["date"], keep="last")

    # =====================================================
    # 4. FEATURE DEFINITIONS
    # =====================================================

    fill_cols = [
        "surprise",
        "surprisePercentage",
        "revenue_growth",
        "net_margin",
        "debt_to_equity",
        "current_ratio",
        "cash_ratio",
        "free_cash_flow",
        "fcf_growth"
    ]

    # ensure all columns exist
    for col in fill_cols:
        if col not in fundamentals.columns:
            fundamentals[col] = np.nan

    # IMPORTANT:
    # only fill WITHIN event space (NOT across fake time)
    fundamentals[fill_cols] = fundamentals[fill_cols].fillna(0)

    # =====================================================
    # 5. EARNINGS EVENT FEATURES
    # =====================================================

    if "earnings_flag" not in fundamentals.columns:
        fundamentals["earnings_flag"] = 0

    fundamentals["earnings_flag"] = (
        fundamentals["earnings_flag"]
        .fillna(0)
        .astype(int)
    )

    fundamentals["event_id"] = (
        fundamentals["date"].diff().dt.days.ne(0).cumsum()
    )

    fundamentals.drop(columns=["event_id"], inplace=True)

    # =====================================================
    # 6. MISSINGNESS FEATURES
    # =====================================================

    for col in fill_cols:
        fundamentals[f"{col}_missing"] = (
            fundamentals[col].isna().astype(int)
        )

    # =====================================================
    # 7. FINAL OUTPUT
    # =====================================================

    fundamentals = fundamentals.sort_values("date")

    return fundamentals

# =========================================================
# RUN
# =========================================================

features = build_feature_set(SYMBOL)

print(features.tail())

# Save
features.to_csv(
    f"{SYMBOL}_fundamental_features.csv"
)


#______________________________NEWS SENTIMENT______________________________
"""
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
"""