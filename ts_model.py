import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler


class XGBModel:

    def __init__(self, n_forecast=3):
        self.n_forecast = n_forecast

        self.model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05
        )

        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.is_trained = False

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------
    def train(self, train_df):

        data = train_df.values

        X = data[:, :-(self.n_forecast+1)]
        Y = data[:, -(self.n_forecast+1):-1]
        print(Y[-1])

        # ✅ Fit scalers ONLY on training data
        self.feature_scaler.fit(X)
        self.target_scaler.fit(Y)

        X_scaled = self.feature_scaler.transform(X)
        Y_scaled = self.target_scaler.transform(Y)

        self.model.fit(X_scaled, Y_scaled)

        self.is_trained = True

    # -------------------------------
    # PREDICT ONE STEP
    # -------------------------------
    def predict(self, feature_row):

        # Convert to array
        data = np.array(feature_row)

        # Split exactly like training
        X = data[:-(self.n_forecast+1)].reshape(1, -1)

        # Scale
        X_scaled = self.feature_scaler.transform(X)

        pred_scaled = self.model.predict(X_scaled)

        # Inverse transform
        pred = self.target_scaler.inverse_transform(pred_scaled)

        return pred[0]

    def validate(self, feature_row):

        if isinstance(feature_row, pd.DataFrame):
            X = feature_row.values
        else:
            X = np.asarray(feature_row)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.feature_scaler.transform(X)

        pred_scaled = self.model.predict(X_scaled)

        pred = self.target_scaler.inverse_transform(pred_scaled)

        return pred[0]