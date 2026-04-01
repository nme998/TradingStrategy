import numpy as np
import pandas as pd

from ts_features import get_feature_data
from ts_model import XGBModel
from ts_backtest import BacktestEngine


def run_walkforward_backtest(
    ticker="AAPL",
    train_start=0.5,   # start with 50% training
    step_size=0.1,     # move forward by 10%
    initial_capital=10000
):

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    data = get_feature_data(ticker)

    n = len(data)
    print("Total dataset size:", n)

    # Convert percentages to indices
    train_end = int(n * train_start)
    step = int(n * step_size)

    engine = BacktestEngine(initial_capital=initial_capital)

    all_predictions = []

    fold = 1

    # -------------------------------
    # WALK-FORWARD LOOP
    # -------------------------------
    while train_end + step <= n:

        test_end = train_end + step

        train_df = data.iloc[:train_end]
        test_df = data.iloc[train_end:test_end]

        print(f"\n--- Fold {fold} ---")
        print(f"Train: 0 → {train_end} ({len(train_df)})")
        print(f"Test: {train_end} → {test_end} ({len(test_df)})")

        # -------------------------------
        # TRAIN MODEL
        # -------------------------------
        model = XGBModel(n_forecast=3)
        model.train(train_df)

        # -------------------------------
        # DAILY SIMULATION
        # -------------------------------
        for idx in range(len(test_df)):

            row = test_df.iloc[idx]

            features = row[:-3]   # exclude targets
            price = row["Close"]
            date = test_df.index[idx]

            # 🔥 PREDICTION
            prediction = model.predict(features)

            all_predictions.append(prediction)

            # 🔥 BACKTEST STEP
            engine.step(
                price=price,
                prediction=prediction,
                date=date
            )

        # -------------------------------
        # MOVE WINDOW FORWARD
        # -------------------------------
        train_end += step
        fold += 1

    print("\nBacktest complete")

    return engine, data, all_predictions