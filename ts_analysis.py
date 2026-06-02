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

    plt.show()

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