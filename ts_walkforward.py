import numpy as np
import pandas as pd
from ts_features import (get_feature_data, add_cross_sectional_features, fit_hmm, apply_hmm)
from ts_model import XGBModel
from ts_backtest import BacktestEngine
from ts_reversal_detection import train_lstm
from ts_strategies.ts_options_vol import OptionsVolatility
from ts_strategies.ts_stat_arb import StatArb
from ts_strategies.ts_main_strat import MainStrat
from ts_options_features import (add_volatility_features, add_volatility_targets)

def calculate_model_confidence(model, X_val, y_val, input_df):
    """
    Calculates model confidence based on historical validation performance.

    Returns:
    - direction_confidence: accuracy of predicting return direction
    - magnitude_confidence: reliability of predicted return magnitude
    - live prediction returns
    """

    # ==========================================
    # Validation predictions
    # ==========================================

    val_predictions = model.predict(X_val)

    # Convert to numpy
    val_predictions = np.array(val_predictions)
    y_val = np.array(y_val)


    # ==========================================
    # 1. Direction Confidence
    # ==========================================

    # Compare signs of predicted vs actual returns
    predicted_direction = np.sign(val_predictions)
    actual_direction = np.sign(y_val)

    direction_accuracy = (
        predicted_direction == actual_direction
    ).mean()

    direction_confidence = direction_accuracy * 100


    # ==========================================
    # 2. Magnitude Confidence
    # ==========================================

    # Absolute return prediction error
    magnitude_error = np.abs(
        val_predictions - y_val
    )

    mean_magnitude_error = magnitude_error.mean()

    # Convert error into confidence
    # Lower error = higher confidence
    magnitude_confidence = max(
        0,
        min(
            100,
            (1 - mean_magnitude_error) * 100
        )
    )


    # ==========================================
    # Live prediction
    # ==========================================

    live_prediction = model.predict(input_df)

    live_prediction = np.array(live_prediction)


    live_direction = np.sign(
        live_prediction
    )


    live_return = live_prediction.mean()


    return {

        # Confidence metrics
        "direction_confidence": direction_confidence,

        "magnitude_confidence": magnitude_confidence,


        # Current prediction
        "predicted_returns": live_prediction,

        "predicted_return_mean": live_return,

        "predicted_direction": live_direction,


        # Debug metrics
        "validation_direction_accuracy": direction_accuracy,

        "mean_prediction_error": mean_magnitude_error
    }


def run_walkforward_backtest(tickers, initial_capital=10000, n_folds=5, lookback=50, rolling_years=5, trading_days_per_year=252):
    raw_data = {}
    for ticker in tickers:
        df = get_feature_data(ticker).copy()
        df = df.sort_index()
        raw_data[ticker] = df

    engine = BacktestEngine(initial_capital=initial_capital, strategy = MainStrat())
    #engine = BacktestEngine(initial_capital=initial_capital, strategy = StatArb(tickers=tickers))
    #engine = BacktestEngine(initial_capital=initial_capital, strategy = OptionsVolatility())

    all_predictions = []

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
            split_idx = int(len(df) * 0.5)
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
        train_df = add_cross_sectional_features(train_df)
        test_df = add_cross_sectional_features(test_df)

        train_vol_df = add_volatility_features(train_df.copy())
        test_vol_df = add_volatility_features(test_df.copy())

        train_vol_df = add_volatility_targets(train_vol_df)
        test_vol_df = add_volatility_targets(test_vol_df)

        # =====================================================
        # CLEAN INFS ONLY
        # =====================================================
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.replace([np.inf, -np.inf], np.nan)

        # =====================================================
        # FORCE COLUMN ORDER
        # =====================================================
        validation_size = 250 * len(tickers)
        target_cols = ["target_1", "target_2", "target_3"]
        feature_cols = [c for c in train_df.columns if c not in (target_cols + ["Ticker"])]

        train_df = train_df[feature_cols + target_cols + ["Ticker"]]
        test_df = test_df[feature_cols + target_cols + ["Ticker"]]
 
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.replace([np.inf, -np.inf], np.nan)

        validation_size = 250 * len(tickers)
        val_df = train_df.tail(validation_size).copy()
        train_df = train_df.iloc[:-validation_size].copy()

        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_cols]

        train_df = train_df.dropna(subset=["target_1", "target_2", "target_3"])

        statarb_history = (pd.concat([train_df, test_df]).sort_index().copy())

        #///////////////////////////////////////////////////////////////////////////////////////////////
        vol_target_cols = ["target_1", "target_2", "target_3"]

        vol_feature_cols = [
            c for c in train_vol_df.columns
            if c not in (vol_target_cols + ["Ticker"])
        ]

        train_vol_df = train_vol_df[
            vol_feature_cols + vol_target_cols + ["Ticker"]
        ]
        #print("TRAIN VOL COLUMNS:", train_vol_df.columns.tolist())
        test_vol_df = test_vol_df[
            vol_feature_cols + vol_target_cols + ["Ticker"]
        ]

        train_vol_df = train_vol_df.replace([np.inf, -np.inf], np.nan)
        test_vol_df = test_vol_df.replace([np.inf, -np.inf], np.nan)

        train_vol_df = train_vol_df.dropna(subset=vol_target_cols)
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # =====================================================
        # TRAIN XGBOOST
        # =====================================================
        print("\nTraining XGBoost...")

        model = XGBModel(n_forecast=3)
        model.train(train_df)

        vol_model = XGBModel(n_forecast=3)
        vol_model.train(train_vol_df)

        val_predictions = []

        for _, row in X_val.iterrows():
            prediction = model.predict(row)

            val_predictions.append({
                "Ticker": row["Ticker"],
                "Date": row.name,
                "prediction": prediction,
                "actual_1": row["target_1"],
                "actual_2": row["target_2"],
                "actual_3": row["target_3"]
            })

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

        tradable_dates = sorted(test_df.iloc[lookback:].index.unique())
        for current_date in tradable_dates:
            date_df = test_df.loc[test_df.index == current_date]
            day_rows = {}
            day_predictions = {}
            day_lookbacks = {}
            vol_lookbacks = {}
            statarb_lookback = {}
            day_vol_predictions = {}

            for _, row in date_df.iterrows():
                ticker = row["Ticker"]
                prediction = model.predict(row)
                confidence = engine.calculate_model_confidence(
                    model=model,
                    X_val=X_val,
                    y_val=y_val,
                    input_df = row[feature_cols].to_frame().T
                )
                print(f"Confidence for {ticker} on {current_date}: {confidence['direction_confidence']} and {confidence['magnitude_confidence']}")

                #/////////////////////////////////////////////////////////////////////////////////////////////////
                vol_row = test_vol_df[
                    (test_vol_df.index == current_date) &
                    (test_vol_df["Ticker"] == ticker)
                ].iloc[0]

                vol_prediction = vol_model.predict(vol_row)

                day_vol_predictions[ticker] = vol_prediction
                #/////////////////////////////////////////////////////////////////////////////////////////////////

                day_rows[ticker] = row
                day_predictions[ticker] = prediction

                lookback_df = test_df[(test_df["Ticker"] == ticker) & (test_df.index < current_date)].tail(lookback)
                vol_lookback_df = test_vol_df[(test_vol_df["Ticker"] == ticker) & (test_vol_df.index < current_date)].tail(lookback)
                day_lookbacks[ticker] = lookback_df
                vol_lookbacks[ticker] = vol_lookback_df

                statarb_df = statarb_history[(statarb_history["Ticker"] == ticker) & (statarb_history.index < current_date)].tail(500)
                statarb_lookback[ticker] = statarb_df

                prediction_debug.append({"Ticker": ticker, "Date": current_date, 
                                         "pred_1": prediction[0], "pred_2": prediction[1], "pred_3": prediction[2]})
                all_predictions.append({"Date": current_date, "Ticker": ticker, "prediction": prediction})

            # ==========================================
            # SINGLE ENGINE CALL PER DATE
            # ==========================================
            if isinstance(engine.strategy, MainStrat):
                engine.step(date=current_date, rows=day_rows, predictions=day_predictions, lookback_windows=day_lookbacks)
            elif isinstance(engine.strategy, StatArb):
                engine.step(date=current_date, rows=day_rows, predictions=day_predictions, lookback_windows=day_lookbacks, stat_arb_lookbacks=statarb_lookback)
            elif isinstance(engine.strategy, OptionsVolatility):
                engine.step(date=current_date, rows=day_rows, predictions=day_predictions, lookback_windows=vol_lookbacks, vol_predictions=day_vol_predictions)
            

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
        dir_acc_1 = np.mean(np.sign(valid_pred_1) == np.sign(actual_1))
        dir_acc_2 = np.mean(np.sign(valid_pred_2) == np.sign(actual_2))
        dir_acc_3 = np.mean(np.sign(valid_pred_3) == np.sign(actual_3))

        print("\n===== DIRECTIONAL ACCURACY =====")
        print(f"Day 1: {dir_acc_1:.4f}")
        print(f"Day 2: {dir_acc_2:.4f}")
        print(f"Day 3: {dir_acc_3:.4f}")

        # =====================================================
        # DISTRIBUTION DEBUG
        # =====================================================
        print("\n===== PRED DISTRIBUTION =====")
        print(f"Pred1 Mean: {np.mean(valid_pred_1):.6f} | Std: {np.std(valid_pred_1):.6f}")
        print(f"Pred2 Mean: {np.mean(valid_pred_2):.6f} | Std: {np.std(valid_pred_2):.6f}")
        print(f"Pred3 Mean: {np.mean(valid_pred_3):.6f} | Std: {np.std(valid_pred_3):.6f}")
        # ==========================================================================================================

    print("\nBacktest complete")
    predictions_df = pd.DataFrame(all_predictions)

    return engine, predictions_df
