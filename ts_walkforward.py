from matplotlib import ticker
import numpy as np
import pandas as pd

from ts_features import apply_lstm, get_feature_data, add_cross_sectional_features
from ts_model import XGBModel
from ts_backtest import BacktestEngine
from ts_features import fit_hmm, fit_lstm, apply_hmm, apply_lstm


def run_walkforward_backtest(
    tickers,
    train_start=0.5,
    step_size=0.1,
    initial_capital=10000
):

    # =========================================================
    # LOAD + STACK MULTI-STOCK DATA
    # =========================================================
    training_data = []
    hmm_models = {}
    lstm_models = {}
    lstm_scalers = {}
    down_states = {}
    up_states = {}
    test_folds = {}
    lookback = 30

    for ticker in tickers:
        df = get_feature_data(ticker)
        split = int(len(df) * 0.5)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        test_df = pd.concat([train_df.iloc[-lookback:], test_df])

        hmm_models[ticker], down_states[ticker], up_states[ticker] = fit_hmm(train_df)
        train_df = apply_hmm(train_df, hmm_models[ticker])
        lstm_models[ticker], lstm_scalers[ticker] = fit_lstm(train_df)
        train_df = apply_lstm(lstm_models[ticker], lstm_scalers[ticker], train_df, lookback=lookback)

        train_df["Ticker"] = ticker

        train_df["target_1"] = train_df["return"].shift(-1)
        train_df["target_2"] = train_df["return"].shift(-2)
        train_df["target_3"] = train_df["return"].shift(-3)

        if len(training_data) == 0:
            training_data = train_df.copy()
        else:
            training_data = pd.concat([training_data, train_df], axis=0)

        # =========================================================
        # WALKFORWARD FOLD PREPARATION
        # =========================================================
        test_len = len(test_df)
        fold_size = test_len // 5

        test_folds[ticker] = []

        for i in range(5):
            start = i * fold_size
            end = (i + 1) * fold_size if i < 4 else test_len

            # ✅ INCLUDE LOOKBACK FROM PREVIOUS DATA
            lookback_start = max(0, start - lookback)

            fold_df = test_df.iloc[lookback_start:end].copy()

            test_folds[ticker].append(fold_df)

    training_data = add_cross_sectional_features(training_data)
    # ===== FORCE COLUMN ORDER (TRAIN) =====
    target_cols = ["target_1", "target_2", "target_3"]

    feature_cols = [c for c in training_data.columns if c not in target_cols + ["Ticker"]]

    training_data = training_data[feature_cols + target_cols + ["Ticker"]]
    # =====================================
    print("Total rows:", len(training_data))
    print("Total unique dates:", training_data.index.unique().shape[0])

    n_dates = len(training_data.index.unique())
    train_end = int(n_dates * 0.5)
    step = int((n_dates * 0.5) / 5)

    # =========================================================
    # BACKTEST ENGINE
    # =========================================================
    engine = BacktestEngine(
        initial_capital=initial_capital,
        hmm_models=hmm_models,
        down_states=down_states,
        up_states=up_states
    )

    # =========================================================
    # WALKFORWARD LOOP
    # =========================================================

    all_predictions = []
    fold = 1

    for fold_idx in range(5):
        print(f"Fold {fold_idx+1} date range:",
            test_df.index.min(),
            "→",
            test_df.index.max())

        processed_chunks = []

        # =====================================================
        # BUILD FULL TEST SET FOR THIS FOLD
        # =====================================================
        for ticker in tickers:

            fold_df = test_folds[ticker][fold_idx].copy()

            fold_df = apply_hmm(fold_df, hmm_models[ticker])
            fold_df = apply_lstm(lstm_models[ticker], lstm_scalers[ticker], fold_df, lookback=lookback)

            # remove lookback leakage
            fold_df = fold_df.iloc[lookback:]

            fold_df["Ticker"] = ticker

            for df_ in [fold_df]:
                df_["target_1"] = df_["return"].shift(-1)
                df_["target_2"] = df_["return"].shift(-2)
                df_["target_3"] = df_["return"].shift(-3)

            processed_chunks.append(fold_df)

        test_df = pd.concat(processed_chunks)

        # IMPORTANT: cross-sectional AFTER merge
        test_df = add_cross_sectional_features(test_df)

        # ===== FORCE COLUMN ORDER =====
        target_cols = ["target_1", "target_2", "target_3"]

        feature_cols = [c for c in test_df.columns if c not in target_cols + ["Ticker"]]

        test_df = test_df[feature_cols + target_cols + ["Ticker"]]
        # ==============================

        # ================= DEBUG BLOCK =================
        print("\n===== DEBUG CHECK =====")

        # 1. Check last row date (sanity: should be recent)
        print("\nLast row of dataset:")
        print(test_df.tail(20))

        print("===== END DEBUG =====\n")
        # =================================================

        # align training slice properly (already built earlier)
        train_df = training_data  # or your prebuilt training_data

        print(f"\n--- Fold {fold_idx + 1} ---")
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        train_df = train_df.dropna(subset=["target_1", "target_2", "target_3"])
        print(train_df.tail(1) )
        # =====================================================
        # TRAIN MODEL
        # =====================================================
        model = XGBModel(n_forecast=3)
        model.train(train_df)

        # =====================================================
        # PREDICTION LOOP (NOW SIMPLE)
        # =====================================================
        for _, row in test_df.iterrows():

            prediction = model.predict(row)
            if row.name in test_df.index[-20:]:
                print(f"Date: {row.name}, Ticker: {row['Ticker']}, Prediction: {prediction}")

            all_predictions.append({
                "Date": row.name,
                "Ticker": row["Ticker"],
                "prediction": prediction
            })

            engine.step(
                ticker=row["Ticker"],
                price=row["Close"],
                high=row["High"],
                low=row["Low"],
                prediction=prediction,
                date=row.name
            )

        fold += 1

    print("\nBacktest complete")

    predictions_df = pd.DataFrame(all_predictions)

    return engine, predictions_df