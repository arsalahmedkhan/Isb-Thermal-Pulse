"""
Isb-Thermal-Pulse: Urban Intervention Simulator
Loads the pre-trained Random Forest model and simulates "What-If"
reforestation scenarios to calculate localized cooling impacts.
"""

import os
import joblib  # Switched from pickle to match model.py saving protocol
import pandas as pd
from config import settings

class ClimateSimulator:
    def __init__(self, model_dir):
        """Loads the trained machine learning model from disk"""
        model_path = os.path.join(model_dir, 'thermal_rf_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")

        # Use joblib directly to open the model weights cleanly
        self.model = joblib.load(model_path)
        print("Trained Random Forest brain successfully loaded into simulator engine.")

    def run_reforestation_simulation(self):
        """
        Simulates an urban heat intervention.
        Compares a heavily paved industrial/commercial sector baseline against
        an aggressive green-canopy planting initiative.
        """
        print("\nConfiguring urban intervention simulation matrix...")

        # Scenario 1: Raw Baseline (e.g., Paved commercial sector / industrial zone)
        baseline_data = pd.DataFrame([{
            'Elevation': 510.0,
            'NDBI': 0.06,
            'NDVI': 0.05
        }])

        # Scenario 2: Reforested Mitigation (The same zone but with an added dense canopy)
        mitigation_data = pd.DataFrame([{
            'Elevation': 510.0,
            'NDBI': -0.02,
            'NDVI': 0.28
        }])

        # CRITICAL FIX: Explicitly re-align columns to match the exact order used during training fit
        feature_order = ['NDVI', 'NDBI', 'Elevation']
        baseline_data = baseline_data[feature_order]
        mitigation_data = mitigation_data[feature_order]

        # Run AI predictions safely now
        predicted_baseline_lst = self.model.predict(baseline_data)[0]
        predicted_mitigated_lst = self.model.predict(mitigation_data)[0]
        cooling_delta = predicted_baseline_lst - predicted_mitigated_lst

        print("\n=========================================")
        print("     URBAN REFORESTATION SIMULATION      ")
        print("=========================================")
        print(f"Baseline Surface Temp (Concrete Grid): {predicted_baseline_lst:.2f}°C")
        print(f"Mitigated Surface Temp (Urban Forest): {predicted_mitigated_lst:.2f}°C")
        print(f"Predicted Local Cooling Impact (ΔT):   -{cooling_delta:.2f}°C")
        print("=========================================")

        if cooling_delta >= 4.0:
            print("SIMULATION RESULT: CRITICAL MITIGATION POTENTIAL ACHIEVED.")
        else:
            print("SIMULATION RESULT: MODERATE MITIGATION POTENTIAL ACHIEVED.")
        # Run AI predictions
        predicted_baseline_lst = self.model.predict(baseline_data)[0]
        predicted_mitigated_lst = self.model.predict(mitigation_data)[0]
        cooling_delta = predicted_baseline_lst - predicted_mitigated_lst

        print("\n=========================================")
        print("     URBAN REFORESTATION SIMULATION      ")
        print("=========================================")
        print(f"Baseline Surface Temp (Concrete Grid): {predicted_baseline_lst:.2f}°C")
        print(f"Mitigated Surface Temp (Urban Forest): {predicted_mitigated_lst:.2f}°C")
        print(f"Predicted Local Cooling Impact (ΔT):   -{cooling_delta:.2f}°C")
        print("=========================================")

        if cooling_delta >= 4.0:
            print("SIMULATION RESULT: CRITICAL MITIGATION POTENTIAL ACHIEVED.")
        else:
            print("SIMULATION RESULT: MODERATE MITIGATION POTENTIAL ACHIEVED.")


if __name__ == "__main__":
    print("Simulator module verified.")