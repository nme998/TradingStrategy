import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Load tradebook
# ----------------------------
df = pd.read_csv("tradebook.csv")


# ----------------------------
# Basic sanity check
# ----------------------------
print("Shape:", df.shape)
print("\nColumns:", df.columns)

# ----------------------------
# Entropy Buckets
# ----------------------------
bins = [0, 0.90, 0.93, 0.95, 1.00]

labels = [
    "<0.90",
    "0.90-0.93",
    "0.93-0.95",
    "0.95-1.00"
]

df["entropy_bucket"] = pd.cut(
    df["entropy"],
    bins=bins,
    labels=labels
)

# ----------------------------
# Overall
# ----------------------------
entropy_overall = (
    df.groupby("entropy_bucket")["pnl"]
    .agg(
        total_trades="count",
        pnl_mean="mean",
        pnl_std="std"
    )
)

# ----------------------------
# Winners
# ----------------------------
entropy_wins = (
    df[df["pnl"] > 0]
    .groupby("entropy_bucket")["pnl"]
    .agg(
        win_count="count",
        win_mean="mean",
        win_std="std"
    )
)

# ----------------------------
# Losers
# ----------------------------
entropy_losses = (
    df[df["pnl"] < 0]
    .groupby("entropy_bucket")["pnl"]
    .agg(
        loss_count="count",
        loss_mean="mean",
        loss_std="std"
    )
)

# ----------------------------
# Combine
# ----------------------------
entropy_stats = (
    entropy_overall
    .join(entropy_wins)
    .join(entropy_losses)
)

# Win rate
entropy_stats["win_rate"] = (
    entropy_stats["win_count"]
    / entropy_stats["total_trades"]
    * 100
)

print("\n--- ENTROPY ANALYSIS ---")
print(entropy_stats)

df["conf_decile"] = pd.qcut(df["confidence"], 10, duplicates="drop")

conf_stats = df.groupby("conf_decile")["pnl"].agg(
    trades="count",
    mean_pnl="mean",
    median_pnl="median",
    total_pnl="sum",
    std="std"
)

conf_stats["win_rate"] = (
    df.groupby("conf_decile")["pnl"]
      .apply(lambda x: (x > 0).mean() * 100)
)

print(conf_stats)

df["score_decile"] = pd.qcut(df["score"], 10, duplicates="drop")

score_stats = df.groupby("score_decile")["pnl"].agg(
    trades="count",
    mean_pnl="mean",
    median_pnl="median",
    total_pnl="sum",
    std="std"
)

score_stats["win_rate"] = (
    df.groupby("score_decile")["pnl"]
      .apply(lambda x: (x > 0).mean() * 100)
)

print(score_stats)

df["exit_date"] = pd.to_datetime(df["exit_date"])
df["month"] = df["exit_date"].dt.month

month_stats = df.groupby("month")["pnl"].agg(
    trades="count",
    mean_pnl="mean",
    total_pnl="sum",
    median_pnl="median"
)

month_stats["win_rate"] = (
    df.groupby("month")["pnl"]
      .apply(lambda x: (x > 0).mean() * 100)
)

print(month_stats.sort_index())

df["entropy_bin"] = pd.qcut(df["entropy"], 4, duplicates="drop")
df["conf_bin"] = pd.qcut(df["confidence"], 4, duplicates="drop")

pivot = pd.pivot_table(
    df,
    values="pnl",
    index="entropy_bin",
    columns="conf_bin",
    aggfunc="mean"
)

print(pivot)

# ----------------------------
# Date buckets
# ----------------------------
df["exit_date"] = pd.to_datetime(df["exit_date"])

df["period_bucket"] = pd.cut(
    df["exit_date"].dt.year,
    bins=[0, 2021, 2022, 2023, 2024, 2025, 9999],
    labels=[
        "<2022",
        "2022-2023",
        "2023-2024",
        "2024-2025",
        "2025-2026",
        ">2026"
    ]
)

# ----------------------------
# Overall stats
# ----------------------------
overall = (
    df.groupby("period_bucket")["pnl"]
    .agg(
        total_trades="count",
        pnl_mean="mean",
        pnl_std="std"
    )
)

# ----------------------------
# Winning trades
# ----------------------------
wins = (
    df[df["pnl"] > 0]
    .groupby("period_bucket")["pnl"]
    .agg(
        win_count="count",
        win_mean="mean",
        win_std="std"
    )
)

# ----------------------------
# Losing trades
# ----------------------------
losses = (
    df[df["pnl"] < 0]
    .groupby("period_bucket")["pnl"]
    .agg(
        loss_count="count",
        loss_mean="mean",
        loss_std="std"
    )
)

# ----------------------------
# Combine
# ----------------------------
year_stats = (
    overall
    .join(wins)
    .join(losses)
)

# =====================================================
# PER STOCK PERFORMANCE
# =====================================================

stock_results = []

for ticker, trades in df.groupby("ticker"):

    trades = trades.sort_values("exit_date").copy()

    # Basic stats
    num_trades = len(trades)
    total_pnl = trades["pnl"].sum()

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]

    num_wins = len(wins)
    num_losses = len(losses)

    win_rate = num_wins / num_trades * 100 if num_trades else 0

    avg_holding = trades["holding_days"].mean()

    # -----------------------------
    # Drawdown
    # -----------------------------
    equity = trades["pnl"].cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_drawdown = abs(drawdown.min())

    # -----------------------------
    # Profit Factor
    # -----------------------------
    gross_profit = wins["pnl"].sum()

    gross_loss = abs(losses["pnl"].sum())

    if gross_loss == 0:
        profit_factor = np.inf
    else:
        profit_factor = gross_profit / gross_loss

    # -----------------------------
    # Annualized Return
    # -----------------------------
    if len(trades) > 1:

        days = (
            trades["exit_date"].max()
            - trades["exit_date"].min()
        ).days

        if days > 0:
            annual_return = total_pnl * (365 / days)
        else:
            annual_return = np.nan
    else:
        annual_return = np.nan

    # -----------------------------
    # Calmar Ratio
    # -----------------------------
    if max_drawdown > 0:
        calmar = annual_return / max_drawdown
    else:
        calmar = np.nan

    stock_results.append({
        "Ticker": ticker,
        "Trades": num_trades,
        "Wins": num_wins,
        "Losses": num_losses,
        "Win Rate (%)": round(win_rate, 2),
        "Total PnL": round(total_pnl, 2),
        "Avg Holding Days": round(avg_holding, 2),
        "Max Drawdown": round(max_drawdown, 2),
        "Profit Factor": round(profit_factor, 2),
        "Calmar": round(calmar, 2) if pd.notna(calmar) else np.nan
    })

stock_stats = (
    pd.DataFrame(stock_results)
      .sort_values("Total PnL", ascending=False)
      .reset_index(drop=True)
)

print("\n========== PER STOCK PERFORMANCE ==========")
print(stock_stats)


print("\n--- YEARLY PNL ANALYSIS ---")
print(year_stats)

df["exit_date"] = pd.to_datetime(df["exit_date"])

wins = df[df["pnl"] > 0]
losses = df[df["pnl"] < 0]

plt.figure(figsize=(14, 6))

plt.bar(
    wins["exit_date"],
    wins["pnl"],
    color="blue",
    width=5,
    label="Positive PnL"
)

plt.bar(
    losses["exit_date"],
    losses["pnl"],
    color="orange",
    width=5,
    label="Negative PnL"
)

plt.axhline(0, color="black", linewidth=1)

plt.xlabel("Date")
plt.ylabel("PnL")
plt.title("Trade PnL by Exit Date")
plt.legend()

plt.tight_layout()
plt.show()

df = df.dropna(subset=["pnl"])

# ----------------------------
# Helper function
# ----------------------------
def plot_relationship(x_col, y_col, title):
    plt.figure(figsize=(10, 5))

    plt.scatter(
        df[x_col],
        df[y_col],
        alpha=0.5
    )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)

    plt.axhline(0, color="black", linewidth=1)

    #plt.show()

    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    print(f"{title} correlation: {corr:.4f}")


for e_bucket in df["entropy_bucket"].unique():
    subset = df[df["entropy_bucket"] == e_bucket]

    print("\nEntropy bucket:", e_bucket)
    print("Score vs PnL corr:",
          subset["score"].corr(subset["pnl"]))
    print("Confidence vs PnL corr:",
          subset["confidence"].corr(subset["pnl"]))

# ----------------------------
# PnL vs Size
# ----------------------------
plot_relationship(
    "size",
    "pnl",
    "PnL vs Position Size"
)

# ----------------------------
# PnL vs Confidence
# ----------------------------
plot_relationship(
    "confidence",
    "pnl",
    "PnL vs Confidence"
)

# ----------------------------
# PnL vs Score
# ----------------------------
plot_relationship(
    "score",
    "pnl",
    "PnL vs Score"
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes = axes.flatten()

for i, bucket in enumerate(labels):

    subset = df[df["entropy_bucket"] == bucket]

    axes[i].hist(
        subset["pnl"],
        bins=30
    )

    axes[i].axvline(
        0,
        color="black",
        linewidth=2
    )

    axes[i].set_title(
        f"Entropy {bucket}\n"
        f"N={len(subset)}  "
        f"Mean={subset['pnl'].mean():.1f}"
    )

    axes[i].set_xlabel("PnL")
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

