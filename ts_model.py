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

        X = data[:, :-self.n_forecast]
        Y = data[:, -self.n_forecast:]

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

        if not self.is_trained:
            raise Exception("Model must be trained before prediction")

        # Ensure correct shape
        X = np.array(feature_row).reshape(1, -1)

        # Scale using TRAIN scalers
        X_scaled = self.feature_scaler.transform(X)

        pred_scaled = self.model.predict(X_scaled)

        # Convert back to real returns
        pred = self.target_scaler.inverse_transform(pred_scaled)

        return pred[0]   # shape: (3,)