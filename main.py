"""
Isb-Thermal-Pulse: Main Execution Entrypoint
Coordinates data ingestion, feature extraction, model orchestration, and simulations.
"""

import os
from config import settings
from src.data_ingestion import DataIngestionPipeline
from src.features import FeatureEngineer
from src.model import ThermalPredictorModel
from src.simulator import ClimateSimulator

def run_pipeline():
    print("=========================================")
    print("     STARTING ISB-THERMAL-PULSE SYSTEM   ")
    print("=========================================\n")

    # 1. Initialize data ingestion
    ingestor = DataIngestionPipeline()
    roi = ingestor.get_islamabad_boundary()
    raw_landsat = ingestor.fetch_landsat_collection(roi)
    raw_dem = ingestor.fetch_elevation_data(roi)

    # 2. Pass raw data into the feature engineering layer
    engineer = FeatureEngineer()
    feature_stack = engineer.process_satellite_features(raw_landsat, raw_dem)

    # 3. Secure download of pixel matrix for training
    raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'islamabad_pixels.csv')
    if not os.path.exists(raw_data_path):
        df = ingestor.sample_features_to_dataframe(feature_stack, roi, num_points=5000)
        df.to_csv(raw_data_path, index=False)
        print(f"Pristine tabular dataset stored at: {raw_data_path}")
    else:
        print(f"Found existing local dataset at {raw_data_path}. Skipping cloud download.")

    # 4. Train the AI Brain
    ml_engine = ThermalPredictorModel()
    # Check if we already have trained weights archived to speed things up
    model_file = os.path.join(settings.PROCESSED_DATA_DIR, 'thermal_rf_model.pkl')
    if not os.path.exists(model_file):
        ml_engine.train_pipeline(raw_data_path)
        ml_engine.save_model(settings.PROCESSED_DATA_DIR)
    else:
        print(f"Found pre-trained model brain weights at {model_file}. Skipping training.")

    # 5. Run Urban Planning "What-If" Simulations
    simulator = ClimateSimulator(settings.PROCESSED_DATA_DIR)
    simulator.run_reforestation_simulation()

    print("\n=========================================")
    print("  SYSTEM EXECUTION COMPLETE: SUCCESS")
    print("=========================================")

if __name__ == "__main__":
    run_pipeline()