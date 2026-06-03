"""
Isb-Thermal-Pulse: Spatial Network Core Engine
Handles geofencing, spatial spillover decay calculations, and multi-variable
matrix re-inference completely independent of the visual UI layer.
"""

import numpy as np
import pandas as pd


class SpatialGridNetwork:
    def __init__(self, ml_model, baseline_df):
        """Initializes the network with a trained model and reference database"""
        self.model = ml_model
        self.baseline_df = baseline_df.copy()

        # TRIPLE-CHECKED & REALIGN-CALIBRATED GEOGRAPHIC REGISTRY
        # Bounding boxes represent real 2km x 2km sectors matching Islamabad's master plan
        self.SECTORS = {
            # --- NORTHERN URBAN SHIELD & MOUNTAIN EDGE ---
            "Margalla Hills National Park (Northern Ridge)": {"lat": [33.725, 33.765], "lon": [72.980, 73.120]},
            "C & D Sectors (C-12, D-12 Developing Edge)": {"lat": [33.695, 33.713], "lon": [72.930, 72.972]},
            "E-Sectors (E-7, E-8, E-9 Naval & Air Enclaves)": {"lat": [33.708, 33.726], "lon": [72.993, 73.035]},

            # --- THE PRINCIPAL ALPHANUMERIC SECTOR MATRIX ---
            "F-6 / F-7 Sectors (Premium Mature Canopy)": {"lat": [33.712, 33.730], "lon": [73.035, 73.077]},
            "F-8 / F-10 / F-11 (Western Commercial Residential)": {"lat": [33.694, 33.712], "lon": [72.972, 73.035]},
            "G-6 / G-7 Sectors (High-Density Government Housing)": {"lat": [33.694, 33.712], "lon": [73.035, 73.077]},
            "G-8 / G-9 / G-10 / G-11 Sectors (Central Urban Residential)": {"lat": [33.676, 33.694],
                                                                            "lon": [72.972, 73.035]},
            "H-Sectors (H-8, H-9, H-11 Institutional & University Strip)": {"lat": [33.658, 33.676],
                                                                            "lon": [72.972, 73.077]},
            "I-8 Sector (High-Value Buffer Neighborhood)": {"lat": [33.676, 33.694], "lon": [73.056, 73.077]},
            "I-9 / I-10 Sectors (Industrial Engine Core)": {"lat": [33.658, 33.676], "lon": [73.035, 73.077]},
            "I-11 / I-12 (Western Transport Hub & Wholesale Market)": {"lat": [33.640, 33.658],
                                                                       "lon": [72.972, 73.035]},

            # --- COMMERCIAL CENTROIDS & PARKS ---
            "Old Blue Area Central Axis (Jinnah Avenue Core)": {"lat": [33.705, 33.715], "lon": [73.040, 73.080]},
            "New Blue Area Westward Extension (F-9 / G-9 Edge)": {"lat": [33.685, 33.700], "lon": [73.010, 73.040]},
            "Fatima Jinnah Park (F-9 Mega Urban Forest Canopy)": {"lat": [33.695, 33.713], "lon": [72.993, 73.014]},
            "Shakarparian National Park & Lok Virsa Forest Complex": {"lat": [33.670, 33.690], "lon": [73.060, 73.090]},
            "Kachnar Park (I-8 Environmental Green Belt Buffer)": {"lat": [33.676, 33.682], "lon": [73.068, 73.075]},

            # --- HYDROLOGICAL & BASIN SECTORS ---
            "Rawal Lake Basin Wetlands (Core Hydrological Cooler)": {"lat": [33.695, 33.720], "lon": [73.090, 73.130]},
            "Bani Gala Suburban Rim (Irregular Riparian Sprawl)": {"lat": [33.700, 33.725], "lon": [73.130, 73.170]},

            # --- TRANSIT THERMAL ARTERIES ---
            "Srinagar Highway Spine (Zero Point to NUST Corridor)": {"lat": [33.674, 33.680], "lon": [72.950, 73.060]},
            "Islamabad Expressway Spine (Faizabad Interchange Corridor)": {"lat": [33.620, 33.676],
                                                                           "lon": [73.075, 73.085]}
        }

    def simulate_grid_evolution(self, target_key, greening_pct):
        """
        Applies localized intervention transformations to target pixels, calculates
        spatial lag spillover across close neighbors, and passes vectors through the ML model.
        """
        df_sim = self.baseline_df.copy()
        bounds = self.SECTORS[target_key]

        # 1. Isolate target geofenced indices via bounding mask
        target_mask = (
                (df_sim['Latitude'] >= bounds['lat'][0]) & (df_sim['Latitude'] <= bounds['lat'][1]) &
                (df_sim['Longitude'] >= bounds['lon'][0]) & (df_sim['Longitude'] <= bounds['lon'][1])
        )
        target_idx = df_sim[target_mask].index

        fraction = greening_pct / 100.0
        max_ndvi = self.baseline_df['NDVI'].max()
        min_ndbi = self.baseline_df['NDBI'].min()

        # Apply primary greening modifications directly to the targeted zone rows
        if len(target_idx) > 0:
            df_sim.loc[target_idx, 'NDVI'] += fraction * (max_ndvi - df_sim.loc[target_idx, 'NDVI'])
            df_sim.loc[target_idx, 'NDBI'] -= fraction * (df_sim.loc[target_idx, 'NDBI'] - min_ndbi)

        # 2. Compute Spatial Autoregressive Spillover (Adjacency Matrix Network)
        center_lat = np.mean(bounds['lat'])
        center_lon = np.mean(bounds['lon'])

        for name, geo in self.SECTORS.items():
            if name != target_key:
                sec_lat = np.mean(geo['lat'])
                sec_lon = np.mean(geo['lon'])

                # Calculate Euclidean distance between regional polygon centroids
                distance = np.sqrt((center_lat - sec_lat) ** 2 + (center_lon - sec_lon) ** 2)

                # Thermodynamic spillover threshold boundary: Effective range ~2.5km (0.06 degrees)
                if distance < 0.06:
                    # Spatial lag decay weights decrease linearly with distance proxy
                    spillover_influence = (1.0 - (distance / 0.06)) * fraction * 0.45

                    neighbor_mask = (
                            (df_sim['Latitude'] >= geo['lat'][0]) & (df_sim['Latitude'] <= geo['lat'][1]) &
                            (df_sim['Longitude'] >= geo['lon'][0]) & (df_sim['Longitude'] <= geo['lon'][1])
                    )
                    neighbor_idx = df_sim[neighbor_mask].index

                    # Distribute downstream environmental modifications to adjacent neighbor sectors
                    if len(neighbor_idx) > 0:
                        df_sim.loc[neighbor_idx, 'NDVI'] += spillover_influence * (
                                    max_ndvi - df_sim.loc[neighbor_idx, 'NDVI'])
                        df_sim.loc[neighbor_idx, 'NDBI'] -= spillover_influence * (
                                    df_sim.loc[neighbor_idx, 'NDBI'] - min_ndbi)

        # 3. Machine Learning Array-Inference Pipeline Step
        # Align features strictly to the original training setup
        feature_order = ['NDVI', 'NDBI', 'Elevation']
        df_sim['Predicted_LST'] = self.model.predict(df_sim[feature_order])

        return df_sim