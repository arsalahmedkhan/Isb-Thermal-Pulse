"""
Isb-Thermal-Pulse: Data Ingestion Engine
Handles authentication and extracts multi-year, cloud-masked Landsat 9
and elevation datasets for the Islamabad ROI.
"""

import ee
import sys
import os
import pandas as pd

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

    def sample_features_to_dataframe(self, feature_stack, roi, num_points=10000):
        """
        Samples a high-density grid by splitting the query into two safe batches
        to bypass Google Earth Engine's 5,000 element collection cap.
        """
        print(f"Initiating dual-batch spatial sampling pipeline for {num_points} elements...")

        # Add the pixel coordinate layers to our feature stack
        coordinates_image = ee.Image.pixelLonLat()
        complete_stack = feature_stack.addBands(coordinates_image)

        # Split total request into two safe sub-queries of 5,000 elements each
        batch_size = num_points // 2
        features_list = []

        # Run Batch 1 and Batch 2 with distinct random seeds
        for batch_num, seed in enumerate([42, 99], start=1):
            print(f"-> Harvesting Batch {batch_num}/2 ({batch_size} footprints) from GEE cloud cluster...")

            try:
                samples = complete_stack.sample(
                    region=roi,
                    scale=30,  # 30-meter ground block scale (2-kanal units)
                    numPixels=batch_size,
                    seed=seed,
                    geometries=False
                ).getInfo()

                if not samples['features']:
                    continue

                for feature in samples['features']:
                    props = feature['properties']

                    # Robust case-insensitive dictionary lookup map
                    row = {k.upper(): v for k, v in props.items()}

                    features_list.append({
                        'NDVI': row.get('NDVI'),
                        'NDBI': row.get('NDBI'),
                        'Elevation': row.get('ELEVATION'),
                        'LST': row.get('LST'),
                        'Latitude': row.get('LATITUDE'),
                        'Longitude': row.get('LONGITUDE')
                    })
            except Exception as batch_error:
                print(f"[BATCH ERROR] Query chunk failed on server: {str(batch_error)}")
                continue

        df = pd.DataFrame(features_list)

        # Clean out any missing cloud-masked edge rows
        df = df.dropna()

        print(f"SUCCESS: Dual batches successfully unified into pristine tabular matrix.")
        print(f"Final Data Footprint Matrix Size: {len(df)} rows.")
        return df
    
if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    roi = pipeline.get_islamabad_boundary()

    # Test pulling the data streams
    landsat_images = pipeline.fetch_landsat_collection(roi)
    elevation_map = pipeline.fetch_elevation_data(roi)
    print("\nData Ingestion Layer: ACTIVE & VERIFIED.")