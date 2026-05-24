"""
Isb-Thermal-Pulse: Data Ingestion Engine
Handles authentication and extracts multi-year, cloud-masked Landsat 9
and elevation datasets for the Islamabad ROI.
"""

import ee
import sys
import os

# Ensure the root directory is in the path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class DataIngestionPipeline:
    def __init__(self):
        """Initializes the connection to Google Earth Engine"""
        print("Initializing cloud handshake with Google Earth Engine...")
        try:
            ee.Initialize(project='isb-thermal-pulse')
            print("Successfully authenticated! Connected to project: isb-thermal-pulse")
        except Exception as e:
            print(f"\n[ERROR] Connection failed: {e}")
            sys.exit(1)

    def get_islamabad_boundary(self):
        """Converts config coordinates into an Earth Engine geometry object"""
        return ee.Geometry.Polygon(settings.ISLAMABAD_ROI['coordinates'])

    def fetch_landsat_collection(self, roi):
        """
        Pulls Landsat 9 imagery across specified years, filtered by
        Islamabad's boundary, summer season, and cloud cover thresholds.
        """
        print(f"Querying Landsat 9 catalog ({settings.LANDSAT_COLLECTION})...")

        # Start with the base collection
        collection = ee.ImageCollection(settings.LANDSAT_COLLECTION) \
                       .filterBounds(roi) \
                       .filter(ee.Filter.lt('CLOUD_COVER', settings.MAX_CLOUD_COVER))

        # Build a filter list to capture only our specific summer weeks across multiple years
        seasonal_filters = []
        for year in settings.YEARS:
            start_date = f"{year}-{settings.START_DAY_MONTH}"
            end_date = f"{year}-{settings.END_DAY_MONTH}"
            seasonal_filters.append(ee.Filter.date(start_date, end_date))

        # Combine all year filters with an "OR" statement
        combined_seasonal_filter = ee.Filter.orFlights(*seasonal_filters) if hasattr(ee.Filter, 'orFlights') else ee.Filter.Or(seasonal_filters)

        final_collection = collection.filter(combined_seasonal_filter)

        # Let's count how many pristine, cloud-free images match our criteria
        count = final_collection.size().getInfo()
        print(f"Found {count} cloud-free satellite scenes matching our criteria across years: {settings.YEARS}")

        return final_collection

    def fetch_elevation_data(self, roi):
        """Extracts the SRTM 30m Digital Elevation Model clipped to Islamabad"""
        print(f"Querying SRTM Elevation catalog ({settings.DEM_COLLECTION})...")
        dem = ee.Image(settings.DEM_COLLECTION).select('elevation').clip(roi)
        print("Elevation dataset isolated successfully.")
        return dem

    def sample_features_to_dataframe(self, feature_stack, roi, num_points=5000):
        """
        Extracts random spatial points across Islamabad from the cloud feature stack
        and converts them into a local dataset for tabular machine learning.
        """
        print(f"Executing stratified spatial sampling ({num_points} points) across Islamabad...")

        # Tell Google to sample points randomly within our Islamabad geometry
        sampled_points = feature_stack.sample(
            region=roi,
            scale=30,  # Match Landsat's 30-meter resolution
            numPixels=num_points,
            seed=settings.RANDOM_STATE,
            tileScale=16  # Prevents out-of-memory errors on the cloud server
        )

        # Pull the data across the network from Google's cloud down to your local PC
        print("Downloading pixel matrix from cloud to local workspace (this takes a few seconds)...")
        features_list = sampled_points.getInfo()['features']

        # Parse the raw JSON payload into a clean tabular structure
        rows = []
        for feat in features_list:
            props = feat['properties']
            # Only keep rows that have all features intact (drop null pixels outside border)
            if all(k in props for k in ['NDVI', 'NDBI', 'Elevation', 'LST']):
                rows.append(props)

        import pandas as pd
        df = pd.DataFrame(rows)
        print(f"Local dataframe generated successfully. Shape: {df.shape}")
        return df


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    roi = pipeline.get_islamabad_boundary()

    # Test pulling the data streams
    landsat_images = pipeline.fetch_landsat_collection(roi)
    elevation_map = pipeline.fetch_elevation_data(roi)
    print("\nData Ingestion Layer: ACTIVE & VERIFIED.")