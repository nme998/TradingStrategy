import numpy as np
import pandas as pd
from ts_features import (
    get_feature_data,
    add_cross_sectional_features,
    fit_hmm,
    apply_hmm
)
from ts_model import XGBModel
from ts_backtest import BacktestEngine
from ts_reversal_detection import train_lstm


def run_walkforward_backtest(
    tickers,
    initial_capital=10000,
    n_folds=5,
    lookback=50,
    rolling_years=5,
    trading_days_per_year=252
):

    raw_data = {}

    for ticker in tickers:

        df = get_feature_data(ticker).copy()

        # preserve ticker-local chronology only
        df = df.sort_index()

        raw_data[ticker] = df

    engine = BacktestEngine(
        initial_capital=initial_capital
    )

    all_predictions = []

    # =========================================================
    # MAIN WALKFORWARD LOOP
    # =========================================================
    for fold_idx in range(n_folds):

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'='*60}")

        training_chunks = []
        testing_chunks = []

        fold_hmm_models = {}
        fold_down_states = {}
        fold_up_states = {}

        fold_lstm_models = {}
        fold_lstm_scalers = {}

        # =====================================================
        # PROCESS EACH TICKER SEPARATELY
        # =====================================================
        for ticker in tickers:

            print(f"\nProcessing {ticker}")

            df = raw_data[ticker].copy()

            # =================================================
            # INITIAL SPLIT
            # =================================================
            split_idx = int(len(df) * 0.5)

            # IMPORTANT:
            # INCLUDE LOOKBACK CONTEXT
            # =================================================
            initial_train = df.iloc[:split_idx]

            future_data = df.iloc[max(0, split_idx - lookback):].copy()

            # =================================================
            # CREATE TEST FOLD
            # =================================================
            future_len = len(future_data)

            fold_size = future_len // n_folds

            test_start = fold_idx * fold_size

            if fold_idx < n_folds - 1:
                test_end = (fold_idx + 1) * fold_size
            else:
                test_end = future_len

            fold_test = future_data.iloc[test_start:test_end].copy()

            # =================================================
            # ROLLING TRAIN WINDOW
            # =================================================
            historical_end = split_idx + test_start

            historical_data = df.iloc[:historical_end].copy()

            rolling_window_size = (rolling_years * trading_days_per_year)

            if len(historical_data) > rolling_window_size:
                fold_train = historical_data.iloc[-rolling_window_size:].copy()

            else:
                fold_train = historical_data.copy()

            print(f"Train rows: {len(fold_train)} | "f"Test rows: {len(fold_test)}")

            print(f"Train range: "f"{fold_train.index.min()} → "f"{fold_train.index.max()}")

            print(f"Test range: "f"{fold_test.index.min()} → "f"{fold_test.index.max()}")

            # =================================================
            # RETRAIN HMM
            # =================================================
            hmm_model, down_state, up_state = fit_hmm(fold_train)

            fold_hmm_models[ticker] = hmm_model
            fold_down_states[ticker] = down_state
            fold_up_states[ticker] = up_state

            fold_train = apply_hmm(fold_train, hmm_model)

            fold_test = apply_hmm(fold_test, hmm_model)

            # =====================================================
            # RETRAIN LSTM
            # =====================================================
            #lstm_model, lstm_scaler = train_lstm(fold_train)

            #fold_lstm_models[ticker] = lstm_model
            #fold_lstm_scalers[ticker] = lstm_scaler

            for dataset in [fold_train, fold_test]:
                dataset["Ticker"] = ticker
                dataset["target_1"] = (dataset["return"].shift(-1))
                dataset["target_2"] = (dataset["return"].shift(-2))
                dataset["target_3"] = (dataset["return"].shift(-3))

            # =================================================
            # STORE
            # =================================================
            training_chunks.append(fold_train)

            testing_chunks.append(fold_test)

        # =====================================================
        # MERGE AFTER ALL FEATURES
        # =====================================================
        train_df = pd.concat(training_chunks)

        test_df = pd.concat(testing_chunks)

        # =====================================================
        # CROSS-SECTIONAL FEATURES
        # =====================================================
        train_df = add_cross_sectional_features(
            train_df
        )

        test_df = add_cross_sectional_features(
            test_df
        )

        # =====================================================
        # CLEAN INFS ONLY
        # =====================================================
        train_df = train_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        test_df = test_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # =====================================================
        # FORCE COLUMN ORDER
        # =====================================================
        target_cols = [
            "target_1",
            "target_2",
            "target_3"
        ]

        feature_cols = [

            c for c in train_df.columns

            if c not in (
                target_cols + ["Ticker"]
            )
        ]

        train_df = train_df[
            feature_cols + target_cols + ["Ticker"]
        ]

        test_df = test_df[
            feature_cols + target_cols + ["Ticker"]
        ]

        train_df = train_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        test_df = test_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        train_df = train_df.dropna(
            subset=[
                "target_1",
                "target_2",
                "target_3"
            ]
        )

        # =====================================================
        # TRAIN XGBOOST
        # =====================================================
        print("\nTraining XGBoost...")

        model = XGBModel(n_forecast=3)

        model.train(train_df)

        # =====================================================
        # UPDATE ENGINE HMM AND LSTM MODELS
        # =====================================================
        engine.hmm_models = fold_hmm_models
        engine.down_states = fold_down_states
        engine.up_states = fold_up_states

        engine.lstm_models = fold_lstm_models
        engine.lstm_scalers = fold_lstm_scalers

        # =====================================================
        # PREDICTION LOOP
        # =====================================================
        print("\nRunning predictions...")
        prediction_debug = []#==========================================================================================================

        tradable_dates = sorted(
            test_df.iloc[lookback:].index.unique()
        )

        for current_date in tradable_dates:

            date_df = test_df.loc[test_df.index == current_date]

            day_rows = {}
            day_predictions = {}
            day_lookbacks = {}

            # ==========================================
            # BUILD ALL TICKER DATA FOR THIS DATE
            # ==========================================

            for _, row in date_df.iterrows():

                ticker = row["Ticker"]

                prediction = model.predict(row)

                day_rows[ticker] = row
                day_predictions[ticker] = prediction

                lookback_df = test_df[
                    (test_df["Ticker"] == ticker)
                    &
                    (test_df.index < current_date)
                ].tail(lookback)

                day_lookbacks[ticker] = lookback_df

                prediction_debug.append({
                    "Ticker": ticker,
                    "Date": current_date,
                    "pred_1": prediction[0],
                    "pred_2": prediction[1],
                    "pred_3": prediction[2]
                })

                all_predictions.append({
                    "Date": current_date,
                    "Ticker": ticker,
                    "prediction": prediction
                })

            # ==========================================
            # SINGLE ENGINE CALL PER DATE
            # ==========================================

            engine.step(
                date=current_date,
                rows=day_rows,
                predictions=day_predictions,
                lookback_windows=day_lookbacks
            )

        print(f"\nFold {fold_idx + 1} complete")

        # ========================================================================================================
        # BUILD PREDICTION DEBUG DATAFRAME
        # =====================================================
        pred_debug_df = pd.DataFrame(prediction_debug)

        # =====================================================
        # COMPUTE REALIZED FUTURE RETURNS
        # =====================================================
        actual_1 = []
        actual_2 = []
        actual_3 = []

        valid_pred_1 = []
        valid_pred_2 = []
        valid_pred_3 = []

        for _, pred_row in pred_debug_df.iterrows():

            ticker = pred_row["Ticker"]

            date = pred_row["Date"]

            raw_df = raw_data[ticker]

            # ensure sorted
            raw_df = raw_df.sort_index()

            if date not in raw_df.index:
                continue

            idx = raw_df.index.get_loc(date)

            # =================================================
            # DAY 1
            # =================================================
            if idx + 1 < len(raw_df):

                realized_1 = raw_df["return"].iloc[idx + 1]

                actual_1.append(realized_1)

                valid_pred_1.append(pred_row["pred_1"])

            # =================================================
            # DAY 2
            # =================================================
            if idx + 2 < len(raw_df):

                realized_2 = raw_df["return"].iloc[idx + 2]

                actual_2.append(realized_2)

                valid_pred_2.append(pred_row["pred_2"])

            # =================================================
            # DAY 3
            # =================================================
            if idx + 3 < len(raw_df):

                realized_3 = raw_df["return"].iloc[idx + 3]

                actual_3.append(realized_3)

                valid_pred_3.append(pred_row["pred_3"])

        # =====================================================
        # SAFE CORRELATION
        # =====================================================
        def safe_corr(a, b):

            if len(a) < 2:
                return np.nan

            if np.std(a) == 0:
                return np.nan

            if np.std(b) == 0:
                return np.nan

            return np.corrcoef(a, b)[0,1]

        # =====================================================
        # CORRELATIONS
        # =====================================================
        corr_1 = safe_corr(valid_pred_1, actual_1)
        corr_2 = safe_corr(valid_pred_2, actual_2)
        corr_3 = safe_corr(valid_pred_3, actual_3)

        print("\n===== PREDICTION QUALITY =====")

        print(f"Fold {fold_idx + 1}")

        print(f"Day 1 Corr: {corr_1:.6f}")
        print(f"Day 2 Corr: {corr_2:.6f}")
        print(f"Day 3 Corr: {corr_3:.6f}")

        # =====================================================
        # DIRECTIONAL ACCURACY
        # =====================================================
        dir_acc_1 = np.mean(
            np.sign(valid_pred_1) ==
            np.sign(actual_1)
        )

        dir_acc_2 = np.mean(
            np.sign(valid_pred_2) ==
            np.sign(actual_2)
        )

        dir_acc_3 = np.mean(
            np.sign(valid_pred_3) ==
            np.sign(actual_3)
        )

        print("\n===== DIRECTIONAL ACCURACY =====")

        print(f"Day 1: {dir_acc_1:.4f}")
        print(f"Day 2: {dir_acc_2:.4f}")
        print(f"Day 3: {dir_acc_3:.4f}")

        # =====================================================
        # DISTRIBUTION DEBUG
        # =====================================================
        print("\n===== PRED DISTRIBUTION =====")

        print(
            f"Pred1 Mean: {np.mean(valid_pred_1):.6f} | "
            f"Std: {np.std(valid_pred_1):.6f}"
        )

        print(
            f"Pred2 Mean: {np.mean(valid_pred_2):.6f} | "
            f"Std: {np.std(valid_pred_2):.6f}"
        )

        print(
            f"Pred3 Mean: {np.mean(valid_pred_3):.6f} | "
            f"Std: {np.std(valid_pred_3):.6f}"
        )
        # ==========================================================================================================

    # =========================================================
    # FINAL OUTPUT
    # =========================================================
    print("\nBacktest complete")

    predictions_df = pd.DataFrame(all_predictions)

    return engine, predictions_df
