"""
Isb-Thermal-Pulse: Data Ingestion Engine
Handles the initial authentication, connection, and cloud queries
to isolate raw Landsat 9 and elevation data for Islamabad.
"""

import ee
import sys
# Reach back into our config directory to pull our settings
sys.path.append('..')
from config import settings

class DataIngestionPipeline:
    def __init__(self):
        """Initializes the connection to Google Earth Engine using our Cloud Project"""
        print("Initializing cloud handshake with Google Earth Engine...")
        try:
            # We explicitly pass our project ID so Google logs it to your quota dashboard
            ee.Initialize(project='isb-thermal-pulse')
            print("Successfully authenticated! Connected to project: isb-thermal-pulse")
        except Exception as e:
            print("\n[ERROR] Connection failed.")
            print("You need to authorize your local PC first. Run 'earthengine authenticate' in your terminal.")
            print(f"Details: {e}")
            sys.exit(1)

    def get_islamabad_boundary(self):
        """Converts our config coordinates into a live Earth Engine geometry object"""
        roi_polygon = settings.ISLAMABAD_ROI
        # Wrap the raw JSON coordinates into a GEE geometry feature
        ee_roi = ee.Geometry.Polygon(roi_polygon['coordinates'])
        return ee_roi

if __name__ == "__main__":
    # Quick internal test to verify the script works independently
    pipeline = DataIngestionPipeline()
    roi = pipeline.get_islamabad_boundary()
    print(f"Successfully generated regional bounding box matrix for Islamabad.")
    print(f"Area footprint verified: {roi.area().divide(1e6).getInfo():.2f} square kilometers.")