"""
Isb-Thermal-Pulse: Main Execution Entrypoint
Coordinates data ingestion, feature extraction, model orchestration, and dual-layer visual analytic compilation.
"""

import os
from config import settings
from src.data_ingestion import DataIngestionPipeline
from src.features import FeatureEngineer
from src.model import ThermalPredictorModel
from src.simulator import ClimateSimulator
from src.viz_layer import ThermalVisualizer
from src.plots import ThermalPlotter # New Import!

def run_pipeline():
    print("=========================================")
    print("     STARTING ISB-THERMAL-PULSE SYSTEM   ")
    print("=========================================\n")

    # 1. Initialize data ingestion
    ingestor = DataIngestionPipeline()
    roi = ingestor.get_islamabad_boundary()
    raw_landsat = ingestor.fetch_landsat_collection(roi)
    raw_dem = ingestor.fetch_elevation_data(roi)

    # 2. Pass data into feature layer
    engineer = FeatureEngineer()
    feature_stack = engineer.process_satellite_features(raw_landsat, raw_dem)

    # 3. Handle tabular CSV cache with True GIS Spatial Poly-Mapping
    raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'islamabad_pixels.csv')
    if not os.path.exists(raw_data_path):
        # A. Pull raw 10,000 pixel batch points from Earth Engine
        df = ingestor.sample_features_to_dataframe(feature_stack, roi, num_points=10000)

        print("\n🌍 Initiating Post-Ingestion GIS Spatial Join...")
        import geopandas as gpd
        from shapely.geometry import Point

        # B. Load your official sector polygon vectors
        geojson_path = os.path.join(settings.DATA_DIR, 'geojson', 'islamabad_sectors.geojson')
        if not os.path.exists(geojson_path):
            print(f"[CRITICAL ERROR] Drop your geometry file at: {geojson_path}")
            sys.exit(1)

        sectors_gdf = gpd.read_file(geojson_path)

        # C. Transform flat rows into live spatial vector dots (WGS84 projection)
        geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
        pixels_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # D. Run Ray-Casting Containment Matrix Join
        # This matches pixels perfectly to irregular boundaries and handles alignment errors
        unified_spatial_df = gpd.sjoin(pixels_gdf, sectors_gdf, how="left", predicate="within")

        # E. Drop the temporary geometry column and save the pristine tagged dataset to disk
        df_final = pd.DataFrame(unified_spatial_df.drop(columns=['geometry', 'index_right']))
        df_final.to_csv(raw_data_path, index=False)
        print("SUCCESS: Every pixel point organically clustered and saved with true sector tags!")
    else:
        print(f"Found existing local dataset at {raw_data_path}. Skipping cloud download.")

    # 4. Handle machine learning model training
    ml_engine = ThermalPredictorModel()
    model_file = os.path.join(settings.PROCESSED_DATA_DIR, 'thermal_rf_model.pkl')
    if not os.path.exists(model_file):
        ml_engine.train_pipeline(raw_data_path)
        ml_engine.save_model(settings.PROCESSED_DATA_DIR)
    else:
        print(f"Found pre-trained model weights. Skipping training.")

    # 5. Run Simulations & Capture All Analytics Output Data Arrays
    simulator = ClimateSimulator(settings.PROCESSED_DATA_DIR)
    hotspot_coords, base_t, miti_t, delta, phases, temps, coolings = simulator.run_reforestation_simulation()

    # 6. Generate the Localized Folium Web Map
    viz = ThermalVisualizer()
    viz.generate_intervention_map(hotspot_coords, base_t, miti_t, delta)

    # 7. Generate the Static Matplotlib Scientific Chart Layer
    plotter = ThermalPlotter()
    plotter.generate_trend_chart(phases, temps, coolings)

    print("\n=========================================")
    print("  SYSTEM EXECUTION COMPLETE: SUCCESS")
    print("=========================================")



if __name__ == "__main__":
    run_pipeline()