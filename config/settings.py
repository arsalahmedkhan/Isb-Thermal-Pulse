"""
Isb-Thermal-Pulse: Configuration Settings
This file acts as the single source of truth for all geographic,
temporal, and model parameters across the pipeline.
"""

import os

# 1. Project Directory Roots (Handles file paths cleanly across any PC)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 2. Geographic Boundaries (Bounding Box for Islamabad Capital Territory)
# Coordinates form a polygon wrapping around the main city sectors and highways
ISLAMABAD_ROI = {
    'type': 'Polygon',
    'coordinates': [[
        [72.85, 33.55],  # Southwest Corner
        [73.25, 33.55],  # Southeast Corner
        [73.25, 33.80],  # Northeast Corner (up into Margallas)
        [72.85, 33.80],  # Northwest Corner
        [72.85, 33.55]   # Close the loop back to start
    ]]
}

# Core point focus for map initializations (Centroid near Zero Point / Srinagar Hwy)
MAP_CENTER = [33.6844, 73.0479]
MAP_ZOOM = 12

# 3. Temporal Windows (Multi-year peak summer targets to capture maximum contrast)
# We will use this to dynamically filter multiple years in our data ingestion engine
START_DAY_MONTH = '05-15'
END_DAY_MONTH = '07-15'
YEARS = [2022, 2023, 2024, 2025]
MAX_CLOUD_COVER = 10


# 4. Satellite Dataset Identifiers (Google Earth Engine Catalog Paths)
LANDSAT_COLLECTION = "LANDSAT/LC09/C02/T1_L2"
DEM_COLLECTION = "USGS/SRTMGL1_003"  # 30m Global Elevation Data

# 5. Model Baseline Parameters
RANDOM_STATE = 42
TRAIN_TEST_SPLIT_RATIO = 0.2