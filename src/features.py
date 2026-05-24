"""
Isb-Thermal-Pulse: Feature Engineering Engine
Calculates environmental indices (NDVI, NDBI) and Land Surface Temperature (LST)
from raw Landsat 9 bands, combining them with elevation data.
"""

import ee


class FeatureEngineer:
    def __init__(self):
        pass

    def process_satellite_features(self, landsat_collection, dem_image):
        """
        Reduces multi-year imagery into a single median composite map
        and calculates LST, NDVI, NDBI, and Elevation features.
        """
        print("Starting geospatial feature engineering pipeline...")

        # 1. Create a Median Composite to remove any transient shadows or noise
        # This gives us the average representation of summer across our years
        median_scene = landsat_collection.median()

        # 2. Calculate NDVI (Vegetation Index)
        # Formula: (NIR - Red) / (NIR + Red) -> Bands 5 and 4 in Landsat 9
        ndvi = median_scene.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

        # 3. Calculate NDBI (Built-Up/Concrete Index)
        # Formula: (SWIR1 - NIR) / (SWIR1 + NIR) -> Bands 6 and 5 in Landsat 9
        ndbi = median_scene.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')

        # 4. Calculate Land Surface Temperature (LST) in Celsius
        # Converts raw thermal infrared data (Band 10) using calibration coefficients
        thermal_band = median_scene.select('ST_B10')
        # Scale factor for Landsat 9 Level-2 Thermal data is 0.00341802, offset is 149.0
        # This calculates temperature in Kelvin, then we subtract 273.15 for Celsius
        lst = thermal_band.multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

        # 5. Combine all layers into a single multi-band master image matrix
        # This stacks our features on top of each other like pancakes
        master_features = ee.Image.cat([
            ndvi,
            ndbi,
            dem_image.rename('Elevation'),
            lst
        ])

        print("Feature stack generated: [NDVI, NDBI, Elevation] -> Target Variable: [LST]")
        return master_features


if __name__ == "__main__":
    # Internal module test to verify code syntax
    print("Feature Engineering layer loaded successfully.")