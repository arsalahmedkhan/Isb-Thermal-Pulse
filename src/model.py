"""
Isb-Thermal-Pulse: Machine Learning Engine
Trains and evaluates a Random Forest Regressor to map the complex,
non-linear relationship between urban features and surface temperatures.
"""

import os
import joblib # Used for saving our trained model weights
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from config import settings


class ThermalPredictorModel:
    def __init__(self):
        """Initializes the model architecture using configuration hyperparameters"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1  # Uses all available CPU cores on your i5 processor for max speed
        )

    def train_pipeline(self, data_path):
        """Loads data, splits it, trains the Random Forest, and logs evaluation metrics"""
        print("Loading local tabular dataset for training...")
        df = pd.read_csv(data_path)

        # Isolate features (X) and target label (y)
        X = df[['NDVI', 'NDBI', 'Elevation']]
        y = df['LST']

        # Split into training and testing pools using our config rules
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=settings.TRAIN_TEST_SPLIT_RATIO,
            random_state=settings.RANDOM_STATE
        )

        print(f"Dataset split completed. Training rows: {X_train.shape[0]} | Testing rows: {X_test.shape[0]}")
        print("Training Random Forest Regressor model (this will be quick)...")

        self.model.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate performance against the test set
        predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        
        print("\n=========================================")
        print("         MODEL PERFORMANCE METRICS       ")
        print("=========================================")
        print(f"R² Score (Variance Explained): {r2:.4f} ({r2 * 100:.1f}%)")
        print(f"RMSE (Average Temp Error):    {rmse:.2f}°C")
        print("=========================================\n")

        # Log Feature Importances to see what drives the heat
        importances = self.model.feature_importances_
        for name, importance in zip(X.columns, importances):
            print(f"Feature Influence -> {name}: {importance * 100:.1f}%")

        return r2, rmse


    def save_model(self, output_dir):
        """Saves the trained weights to disk for future simulation predictions"""
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, 'thermal_rf_model.pkl')

        # Use joblib to dump the model weights cleanly
        import joblib
        joblib.dump(self.model, model_save_path)

        print(f"\nTrained model brain safely archived at: {model_save_path}")