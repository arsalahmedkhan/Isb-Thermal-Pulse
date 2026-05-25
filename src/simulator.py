"""
Isb-Thermal-Pulse: Urban Intervention Simulator
Contains both the flawed manual guesswork scenario (Negative Test Case)
and the advanced empirical linear interpolation pipeline for diagnostic comparison.
"""

import os
import joblib
import pandas as pd
from config import settings

class ClimateSimulator:
    def __init__(self, model_dir):
        """Loads the trained machine learning model from disk"""
        model_path = os.path.join(model_dir, 'thermal_rf_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}.")

        self.model = joblib.load(model_path)
        print("Trained Random Forest brain successfully loaded into simulator engine.")

    def run_flawed_manual_simulation(self):
        """
        [NEGATIVE TEST CASE] Demonstrates the 'Garbage In, Garbage Out' trap.
        Feeds the model an unphysical, hand-guessed hybrid feature combination.
        """
        print("\n=========================================")
        print("   DIAGNOSTIC CRASH TEST: FLAWED METHOD  ")
        print("=========================================")
        print("CRITIQUE: Feeding hand-guessed numbers that break natural correlations...")

        # Creating an unphysical 'chimera' pixel (High concrete AND high trees simultaneously)
        baseline_data = pd.DataFrame([{'Elevation': 510.0, 'NDBI': 0.06, 'NDVI': 0.05}])
        flawed_mitigation = pd.DataFrame([{'Elevation': 510.0, 'NDBI': 0.05, 'NDVI': 0.18}])

        feature_order = ['NDVI', 'NDBI', 'Elevation']
        baseline_data = baseline_data[feature_order]
        flawed_mitigation = flawed_mitigation[feature_order]

        pred_base = self.model.predict(baseline_data)[0]
        pred_miti = self.model.predict(flawed_mitigation)[0]
        delta = pred_base - pred_miti

        print(f"Baseline Surface Temp:  {pred_base:.2f}°C")
        print(f"Flawed Mitigated Temp:  {pred_miti:.2f}°C")
        print(f"Calculated Delta (ΔT):   -{delta:.2f}°C  <-- !! POSITIVE HEAT SPIKE !!")
        print("\nWHY DID THIS FAIL?")
        print("The Random Forest has never seen high NDBI paired with high NDVI at this altitude.")
        print("Without natural data correlation, the model falls out-of-bounds and hallucinates.")
        print("=========================================\n")

    def run_empirical_phase_simulation(self):
        """
        [PRODUCTION METHOD] Runs a realistic urban intervention by walking
        proportionally between two real, existing data coordinates.
        """
        raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'islamabad_pixels.csv')
        df = pd.read_csv(raw_data_path)

        # Isolate flat valley floor sectors (around 500m elevation)
        plateau_df = df[(df['Elevation'] >= 490) & (df['Elevation'] <= 530)]

        # Extract our two real empirical anchor points
        hotspot = plateau_df.sort_values(by='NDBI', ascending=False).iloc[0]
        green_target = plateau_df.sort_values(by='NDVI', ascending=False).iloc[0]

        print("=========================================")
        print("  PRODUCTION WORKFLOW: EMPIRICAL BLEND   ")
        print("=========================================")
        print(f"Targeting Hotspot Pixel at Elevation: {hotspot['Elevation']:.1f}m")
        print(f"True Baseline Satellite Temp:         {hotspot['LST']:.2f}°C")
        print("-----------------------------------------")

        feature_order = ['NDVI', 'NDBI', 'Elevation']

        for percent in [0, 20, 40, 60, 80, 100]:
            fraction = percent / 100.0
            simulated_ndvi = hotspot['NDVI'] + fraction * (green_target['NDVI'] - hotspot['NDVI'])
            simulated_ndbi = hotspot['NDBI'] + fraction * (green_target['NDBI'] - hotspot['NDBI'])

            sim_row = pd.DataFrame([{
                'NDVI': simulated_ndvi,
                'NDBI': simulated_ndbi,
                'Elevation': hotspot['Elevation']
            }])[feature_order]

            predicted_temp = self.model.predict(sim_row)[0]
            current_cooling = hotspot['LST'] - predicted_temp

            phase_name = "BASELINE" if percent == 0 else f"PHASE {percent//20} ({percent}%)"
            print(f"{phase_name:<12} -> NDVI: {simulated_ndvi:.3f} | NDBI: {simulated_ndbi:.3f} | Predicted Temp: {predicted_temp:.2f}°C (Cooling: -{current_cooling:.2f}°C)")

        print("=========================================\n")

    def run_reforestation_simulation(self):
        """Orchestrates both scenarios sequentially for presentation comparison"""
        self.run_flawed_manual_simulation()
        self.run_empirical_phase_simulation()

if __name__ == "__main__":
    print("Simulator diagnostic suite active.")