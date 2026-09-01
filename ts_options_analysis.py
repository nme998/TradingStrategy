import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

TRADEBOOK_PATH = "tradebook.csv"

# Current strategy thresholds
PROFIT_TARGET = 0.40
THETA_EXIT_THRESHOLD = 0.04
EXPIRY_EXIT_DAYS = 5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRADEBOOK_PATH)

print("=" * 70)
print("OPTIONS STRATEGY ANALYSIS")
print("=" * 70)

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "trade_id",
    "option_type",
    "entry_price",
    "premium_paid",
    "exit_price",
    "premium_received",
    "size",
    "direction",
    "pnl",
    "entry_score",
    "time_to_expiry",
    "entry_date",
    "exit_date",
    "is_open",
    "strike",
    "expiry",
    "underlying_entry",
    "underlying_exit",
    "implied_vol",
    "predicted_vol",
    "exit_implied_vol",
    "exit_predicted_vol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "max_risk",
    "confidence",
    "regime",
    "ticker",
    "current_price"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print("\nWARNING - Missing columns:")
    print(missing)


# ============================================================
# DATA CLEANING
# ============================================================

numeric_columns = [
    "entry_price",
    "premium_paid",
    "exit_price",
    "premium_received",
    "size",
    "pnl",
    "entry_score",
    "time_to_expiry",
    "strike",
    "underlying_entry",
    "underlying_exit",
    "implied_vol",
    "predicted_vol",
    "exit_implied_vol",
    "exit_predicted_vol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "max_risk",
    "confidence",
    "current_price"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


df["entry_date"] = pd.to_datetime(
    df["entry_date"],
    errors="coerce"
)

df["exit_date"] = pd.to_datetime(
    df["exit_date"],
    errors="coerce"
)

df = df.dropna(subset=["pnl"])

print("\nValid trades:", len(df))


# ============================================================
# DATA SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("1. DATA SANITY CHECK")
print("=" * 70)

print("\nDuplicate trade IDs:",
      df["trade_id"].duplicated().sum())

print("Missing PnL:",
      df["pnl"].isna().sum())

print("Zero/negative option entry prices:",
      (df["entry_price"] <= 0).sum())

print("Zero/negative position sizes:",
      (df["size"] <= 0).sum())

print("Negative premium paid:",
      (df["premium_paid"] < 0).sum())


# ============================================================
# DERIVED VARIABLES
# ============================================================

# Option return
df["option_return"] = np.where(
    df["premium_paid"] != 0,
    df["pnl"] / df["premium_paid"],
    np.nan
)


# Underlying return
df["underlying_return"] = np.where(
    df["underlying_entry"] != 0,
    (
        df["underlying_exit"]
        - df["underlying_entry"]
    ) / df["underlying_entry"],
    np.nan
)


# Absolute volatility edge
df["vol_edge"] = (
    df["predicted_vol"]
    - df["implied_vol"]
)


# Relative volatility edge
df["vol_edge_pct"] = np.where(
    df["implied_vol"] != 0,
    (
        df["predicted_vol"]
        - df["implied_vol"]
    ) / df["implied_vol"],
    np.nan
)


# Change in implied volatility
df["iv_change"] = (
    df["exit_implied_vol"]
    - df["implied_vol"]
)


# Change in predicted volatility
df["predicted_vol_change"] = (
    df["exit_predicted_vol"]
    - df["predicted_vol"]
)


# Absolute delta
df["abs_delta"] = df["delta"].abs()


# Theta as a fraction of option price
df["theta_ratio"] = np.where(
    df["current_price"] != 0,
    df["theta"].abs() / df["current_price"],
    np.nan
)


# Exit return
df["exit_return"] = np.where(
    df["entry_price"] != 0,
    (
        df["exit_price"]
        - df["entry_price"]
    ) / df["entry_price"],
    np.nan
)


# Holding period
df["holding_days"] = (
    df["exit_date"] - df["entry_date"]
).dt.days


# Calendar information
df["entry_year"] = df["entry_date"].dt.year
df["exit_year"] = df["exit_date"].dt.year
df["entry_month"] = df["entry_date"].dt.month
df["exit_month"] = df["exit_date"].dt.month


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def performance_table(data):
    """
    Generate a compact performance table.
    """

    if len(data) == 0:
        return pd.Series({
            "Trades": 0,
            "Win Rate (%)": np.nan,
            "Mean PnL": np.nan,
            "Median PnL": np.nan,
            "Total PnL": np.nan,
            "Avg Win": np.nan,
            "Avg Loss": np.nan,
            "Profit Factor": np.nan,
            "Mean Return (%)": np.nan
        })

    wins = data[data["pnl"] > 0]
    losses = data[data["pnl"] < 0]

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())

    if gross_loss == 0:
        profit_factor = np.inf
    else:
        profit_factor = gross_profit / gross_loss

    return pd.Series({
        "Trades": len(data),

        "Win Rate (%)":
            (data["pnl"] > 0).mean() * 100,

        "Mean PnL":
            data["pnl"].mean(),

        "Median PnL":
            data["pnl"].median(),

        "Total PnL":
            data["pnl"].sum(),

        "Avg Win":
            wins["pnl"].mean()
            if len(wins) else np.nan,

        "Avg Loss":
            losses["pnl"].mean()
            if len(losses) else np.nan,

        "Profit Factor":
            profit_factor,

        "Mean Return (%)":
            data["option_return"].mean() * 100
    })


def grouped_analysis(data, column):
    """
    Performance by categorical/bucket column.
    """

    result = (
        data.groupby(column, observed=True)
        .apply(performance_table)
    )

    return result


def correlation_analysis(data, columns):
    """
    Correlation between variables and PnL.
    """

    available = [
        col for col in columns
        if col in data.columns
    ]

    print("\nPnL correlations:")

    for col in available:

        subset = data[[col, "pnl"]].dropna()

        if len(subset) < 2:
            continue

        corr = subset[col].corr(subset["pnl"])

        print(
            f"{col:25s}: {corr: .4f}"
        )


# ============================================================
# 2. CALL VS PUT
# ============================================================

print("\n" + "=" * 70)
print("2. CALL VS PUT")
print("=" * 70)

call_put_stats = grouped_analysis(
    df,
    "option_type"
)

print(call_put_stats)


# ============================================================
# 3. VOLATILITY EDGE
# ============================================================

print("\n" + "=" * 70)
print("3. VOLATILITY EDGE")
print("=" * 70)

print("""
vol_edge = predicted_vol - implied_vol

Positive:
    Model predicts higher volatility than market-implied volatility.

Negative:
    Model predicts lower volatility than market-implied volatility.
""")


df["vol_edge_bucket"] = pd.cut(
    df["vol_edge_pct"],
    bins=[
        -np.inf,
        -0.20,
        -0.10,
        -0.05,
        0,
        0.05,
        0.10,
        0.20,
        np.inf
    ],
    labels=[
        "< -20%",
        "-20% to -10%",
        "-10% to -5%",
        "-5% to 0%",
        "0% to 5%",
        "5% to 10%",
        "10% to 20%",
        "> 20%"
    ]
)

vol_edge_stats = grouped_analysis(
    df,
    "vol_edge_bucket"
)

print(vol_edge_stats)


# ============================================================
# 4. PREDICTED VS IMPLIED VOLATILITY
# ============================================================

print("\n" + "=" * 70)
print("4. VOLATILITY PREDICTION ANALYSIS")
print("=" * 70)

correlation_analysis(
    df,
    [
        "predicted_vol",
        "implied_vol",
        "vol_edge",
        "vol_edge_pct",
        "iv_change",
        "predicted_vol_change"
    ]
)

print("\nVolatility summary:")

print(
    df[
        [
            "implied_vol",
            "predicted_vol",
            "vol_edge",
            "vol_edge_pct",
            "exit_implied_vol",
            "exit_predicted_vol"
        ]
    ].describe()
)


# ============================================================
# 5. VOLATILITY EDGE VS REALISED PNL
# ============================================================

print("\n" + "=" * 70)
print("5. VOLATILITY EDGE BY WIN/LOSS")
print("=" * 70)

print(
    df.groupby(
        df["pnl"] > 0
    )[
        [
            "vol_edge_pct",
            "predicted_vol",
            "implied_vol"
        ]
    ].mean()
)


# ============================================================
# 6. REGIME ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6. REGIME ANALYSIS")
print("=" * 70)

regime_stats = grouped_analysis(
    df,
    "regime"
)

print(regime_stats)


# ============================================================
# 7. GREEKS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. GREEKS ANALYSIS")
print("=" * 70)

correlation_analysis(
    df,
    [
        "delta",
        "abs_delta",
        "gamma",
        "theta",
        "vega",
        "theta_ratio"
    ]
)


# ------------------------------------------------------------
# Delta buckets
# ------------------------------------------------------------

df["delta_bucket"] = pd.cut(
    df["abs_delta"],
    bins=[
        0,
        0.20,
        0.40,
        0.60,
        0.80,
        1.00,
        np.inf
    ],
    labels=[
        "0-0.20",
        "0.20-0.40",
        "0.40-0.60",
        "0.60-0.80",
        "0.80-1.00",
        ">1.00"
    ]
)

print("\nDelta analysis:")

print(
    grouped_analysis(
        df,
        "delta_bucket"
    )
)


# ------------------------------------------------------------
# Gamma buckets
# ------------------------------------------------------------

df["gamma_bucket"] = pd.qcut(
    df["gamma"],
    4,
    duplicates="drop"
)

print("\nGamma quartiles:")

print(
    grouped_analysis(
        df,
        "gamma_bucket"
    )
)


# ------------------------------------------------------------
# Vega buckets
# ------------------------------------------------------------

df["vega_bucket"] = pd.qcut(
    df["vega"],
    4,
    duplicates="drop"
)

print("\nVega quartiles:")

print(
    grouped_analysis(
        df,
        "vega_bucket"
    )
)


# ============================================================
# 8. THETA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("8. THETA ANALYSIS")
print("=" * 70)

df["theta_ratio_bucket"] = pd.cut(
    df["theta_ratio"],
    bins=[
        0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        np.inf
    ],
    labels=[
        "<1%",
        "1-2%",
        "2-3%",
        "3-4%",
        "4-5%",
        "5-7.5%",
        "7.5-10%",
        ">10%"
    ]
)

theta_stats = grouped_analysis(
    df,
    "theta_ratio_bucket"
)

print(theta_stats)

print(
    "\nCurrent theta exit threshold:",
    THETA_EXIT_THRESHOLD
)


# ============================================================
# 9. TIME TO EXPIRY
# ============================================================

print("\n" + "=" * 70)
print("9. TIME TO EXPIRY")
print("=" * 70)

df["expiry_bucket"] = pd.cut(
    df["time_to_expiry"],
    bins=[
        -np.inf,
        15 / 252,
        30 / 252,
        60 / 252,
        90 / 252,
        180 / 252,
        np.inf
    ],
    labels=[
        "<15 days",
        "15-30 days",
        "30-60 days",
        "60-90 days",
        "90-180 days",
        ">180 days"
    ]
)

print(
    grouped_analysis(
        df,
        "expiry_bucket"
    )
)


# ============================================================
# 10. UNDERLYING MOVEMENT
# ============================================================

print("\n" + "=" * 70)
print("10. UNDERLYING MOVEMENT")
print("=" * 70)

correlation_analysis(
    df,
    [
        "underlying_return",
        "option_return",
        "vol_edge_pct"
    ]
)

df["underlying_return_bucket"] = pd.cut(
    df["underlying_return"],
    bins=[
        -np.inf,
        -0.10,
        -0.05,
        -0.02,
        0,
        0.02,
        0.05,
        0.10,
        np.inf
    ],
    labels=[
        "< -10%",
        "-10% to -5%",
        "-5% to -2%",
        "-2% to 0%",
        "0% to 2%",
        "2% to 5%",
        "5% to 10%",
        ">10%"
    ]
)

print("\nUnderlying return buckets:")

print(
    grouped_analysis(
        df,
        "underlying_return_bucket"
    )
)


# ============================================================
# 11. POSITION SIZE / PREMIUM ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("11. POSITION SIZE / PREMIUM")
print("=" * 70)

correlation_analysis(
    df,
    [
        "size",
        "premium_paid",
        "max_risk",
        "confidence",
        "entry_score"
    ]
)

df["premium_bucket"] = pd.qcut(
    df["premium_paid"],
    4,
    duplicates="drop"
)

print("\nPremium quartiles:")

print(
    grouped_analysis(
        df,
        "premium_bucket"
    )
)


# ============================================================
# 12. TICKER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("12. PER-TICKER PERFORMANCE")
print("=" * 70)

ticker_stats = grouped_analysis(
    df,
    "ticker"
)

ticker_stats = ticker_stats.sort_values(
    "Total PnL",
    ascending=False
)

print(ticker_stats)


# ============================================================
# 13. YEARLY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("13. YEARLY PERFORMANCE")
print("=" * 70)

year_stats = grouped_analysis(
    df,
    "exit_year"
)

print(year_stats)


# ============================================================
# 14. MONTHLY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("14. MONTHLY PERFORMANCE")
print("=" * 70)

month_stats = grouped_analysis(
    df,
    "exit_month"
)

print(month_stats)


# ============================================================
# 15. OPTION RETURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("15. OPTION RETURN")
print("=" * 70)

print(
    df["option_return"]
    .describe()
)

print("\nReturn by option type:")

print(
    df.groupby("option_type")["option_return"]
      .agg([
          "count",
          "mean",
          "median",
          "std",
          "min",
          "max"
      ])
)


# ============================================================
# 16. EXIT PROFIT TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("16. PROFIT TARGET ANALYSIS")
print("=" * 70)

df["profit_target_region"] = pd.cut(
    df["option_return"],
    bins=[
        -np.inf,
        0,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.75,
        1.00,
        np.inf
    ],
    labels=[
        "<0%",
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-75%",
        "75-100%",
        ">100%"
    ]
)

print(
    grouped_analysis(
        df,
        "profit_target_region"
    )
)

print(
    "\nCurrent profit target:",
    PROFIT_TARGET * 100,
    "%"
)


# ============================================================
# 17. INTERACTION: VOL EDGE × REGIME
# ============================================================

print("\n" + "=" * 70)
print("17. VOL EDGE × REGIME")
print("=" * 70)

pivot = pd.pivot_table(
    df,
    values="pnl",
    index="vol_edge_bucket",
    columns="regime",
    aggfunc="mean"
)

print("\nMean PnL:")
print(pivot)


pivot_return = pd.pivot_table(
    df,
    values="option_return",
    index="vol_edge_bucket",
    columns="regime",
    aggfunc="mean"
)

print("\nMean option return:")
print(pivot_return)


# ============================================================
# 18. INTERACTION: VOL EDGE × CALL/PUT
# ============================================================

print("\n" + "=" * 70)
print("18. VOL EDGE × OPTION TYPE")
print("=" * 70)

pivot = pd.pivot_table(
    df,
    values="pnl",
    index="vol_edge_bucket",
    columns="option_type",
    aggfunc="mean"
)

print(pivot)


# ============================================================
# 19. INTERACTION: DELTA × VOL EDGE
# ============================================================

print("\n" + "=" * 70)
print("19. DELTA × VOL EDGE")
print("=" * 70)

pivot = pd.pivot_table(
    df,
    values="pnl",
    index="delta_bucket",
    columns="vol_edge_bucket",
    aggfunc="mean"
)

print(pivot)


# ============================================================
# 20. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("20. CORRELATION MATRIX")
print("=" * 70)

correlation_columns = [
    "pnl",
    "option_return",
    "underlying_return",
    "vol_edge",
    "vol_edge_pct",
    "predicted_vol",
    "implied_vol",
    "iv_change",
    "predicted_vol_change",
    "delta",
    "gamma",
    "theta",
    "theta_ratio",
    "vega",
    "size",
    "premium_paid",
    "max_risk",
    "time_to_expiry"
]

available = [
    col for col in correlation_columns
    if col in df.columns
]

print(
    df[available].corr()["pnl"]
    .sort_values(ascending=False)
)


# ============================================================
# 21. ATM / OTM ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("21. ATM / OTM ANALYSIS")
print("=" * 70)

if "selection" in df.columns:

    print(
        grouped_analysis(
            df,
            "selection"
        )
    )

else:

    print(
        """
SELECTION COLUMN NOT FOUND.

The current tradebook does not contain whether the
contract was ATM or OTM.

Add the following to OptionTrade:

    self.selection = None

Then when creating the trade:

    trade.selection = contract.get("selection", "ATM")

And make sure your tradebook writer includes:

    selection

Once that is added, this analysis will automatically run.
"""
    )


# ============================================================
# 22. EXIT REASON ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("22. EXIT REASON ANALYSIS")
print("=" * 70)

if "exit_reason" in df.columns:

    print(
        grouped_analysis(
            df,
            "exit_reason"
        )
    )

else:

    print(
        """
EXIT REASON COLUMN NOT FOUND.

Your strategy currently has:

    EXIT_SCORE
    PROFIT_TARGET
    THETA_EXIT
    EXPIRY

but exit_reason is not currently recorded in the tradebook.

Adding it would allow us to determine which exit
mechanism is actually contributing the most PnL.
"""
    )


# ============================================================
# 23. THRESHOLD DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("23. THRESHOLD DIAGNOSTICS")
print("=" * 70)


# Profit target
near_profit_target = df[
    (df["option_return"] >= 0.30) &
    (df["option_return"] <= 0.50)
]

print("\nTrades around 40% profit target:")

print(
    performance_table(
        near_profit_target
    )
)


# Theta threshold
near_theta = df[
    (df["theta_ratio"] >= 0.025) &
    (df["theta_ratio"] <= 0.06)
]

print("\nTrades around 4% theta threshold:")

print(
    performance_table(
        near_theta
    )
)


# ============================================================
# 24. MOST IMPORTANT CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("24. KEY RELATIONSHIPS")
print("=" * 70)

key_relationships = [
    ("Volatility edge", "vol_edge_pct"),
    ("Predicted volatility", "predicted_vol"),
    ("Implied volatility", "implied_vol"),
    ("Underlying return", "underlying_return"),
    ("Delta", "delta"),
    ("Gamma", "gamma"),
    ("Theta", "theta"),
    ("Theta ratio", "theta_ratio"),
    ("Vega", "vega"),
    ("Position size", "size"),
    ("Premium paid", "premium_paid")
]

for name, column in key_relationships:

    subset = df[[column, "pnl"]].dropna()

    if len(subset) < 2:
        continue

    corr = subset[column].corr(
        subset["pnl"]
    )

    print(
        f"{name:25s}: {corr: .4f}"
    )


# ============================================================
# 25. PLOTS
# ============================================================


def scatter_plot(x, y, title):

    subset = df[[x, y]].dropna()

    if len(subset) == 0:
        return

    plt.figure(figsize=(10, 6))

    plt.scatter(
        subset[x],
        subset[y],
        alpha=0.5
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# PnL vs volatility edge
# ------------------------------------------------------------

scatter_plot(
    "vol_edge_pct",
    "pnl",
    "PnL vs Relative Volatility Edge"
)


# ------------------------------------------------------------
# PnL vs underlying return
# ------------------------------------------------------------

scatter_plot(
    "underlying_return",
    "pnl",
    "PnL vs Underlying Return"
)


# ------------------------------------------------------------
# PnL vs delta
# ------------------------------------------------------------

scatter_plot(
    "abs_delta",
    "pnl",
    "PnL vs Absolute Delta"
)


# ------------------------------------------------------------
# PnL vs theta ratio
# ------------------------------------------------------------

scatter_plot(
    "theta_ratio",
    "pnl",
    "PnL vs Theta / Option Price"
)


# ------------------------------------------------------------
# PnL vs vega
# ------------------------------------------------------------

scatter_plot(
    "vega",
    "pnl",
    "PnL vs Vega"
)


# ------------------------------------------------------------
# PnL vs gamma
# ------------------------------------------------------------

scatter_plot(
    "gamma",
    "pnl",
    "PnL vs Gamma"
)


# ------------------------------------------------------------
# Predicted vs implied volatility
# ------------------------------------------------------------

subset = df[
    [
        "predicted_vol",
        "implied_vol"
    ]
].dropna()

if len(subset) > 0:

    plt.figure(figsize=(8, 8))

    plt.scatter(
        subset["implied_vol"],
        subset["predicted_vol"],
        alpha=0.5
    )

    min_value = min(
        subset["implied_vol"].min(),
        subset["predicted_vol"].min()
    )

    max_value = max(
        subset["implied_vol"].max(),
        subset["predicted_vol"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Implied Volatility")
    plt.ylabel("Predicted Volatility")
    plt.title(
        "Predicted Volatility vs Implied Volatility"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 26. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("26. DIAGNOSTIC SUMMARY")
print("=" * 70)

print(
    "\nOverall performance:"
)

print(
    performance_table(df)
)


print("\nHighest PnL tickers:")

print(
    ticker_stats[
        ["Trades", "Win Rate (%)", "Total PnL", "Profit Factor"]
    ]
    .head(10)
)


print("\nLowest PnL tickers:")

print(
    ticker_stats[
        ["Trades", "Win Rate (%)", "Total PnL", "Profit Factor"]
    ]
    .tail(10)
)


print("\nLargest positive volatility edges:")

print(
    df.nlargest(
        10,
        "vol_edge_pct"
    )[
        [
            "ticker",
            "option_type",
            "vol_edge_pct",
            "pnl",
            "option_return"
        ]
    ]
)


print("\nLargest negative volatility edges:")

print(
    df.nsmallest(
        10,
        "vol_edge_pct"
    )[
        [
            "ticker",
            "option_type",
            "vol_edge_pct",
            "pnl",
            "option_return"
        ]
    ]
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)