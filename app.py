"""
Isb-Thermal-Pulse: Advanced Map Studio (Sector-Grid Core Matrix)
Features dynamic city-wide sector selection, neighborhood spatial spillover
thermodynamics, and stabilized map layers utilizing folium_static. - Fahad Qaseem
""" 

import os
import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import numpy as np
from config import settings

# 1. Page Configuration
st.set_page_config(
    page_title="Isb-Thermal-Pulse Grid Studio",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ Isb-Thermal-Pulse: Sector-Grid Spatial Simulator")
st.markdown("### City-Scale Multi-Sector Policy Analysis & Thermal Spillover Waves")
st.write("---")

# 2. Data Loading Optimization
@st.cache_resource
def load_ai_brain():
    model_path = os.path.join(settings.PROCESSED_DATA_DIR, 'thermal_rf_model.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_empirical_data():
    raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'islamabad_pixels.csv')
    return pd.read_csv(raw_data_path)

try:
    rf_model = load_ai_brain()
    df_pixels = load_empirical_data()
except Exception as e:
    st.error("🚨 Configuration Error: Verify your local dataset cache!")
    st.stop()

# TRIPLE-CHECKED INDUSTRIAL MASTER PLAN GRID BOUNDS
SECTORS = {
    "Margalla Hills National Park (Northern Ridge)": {"lat": [33.725, 33.765], "lon": [72.980, 73.120]},
    "C & D Sectors (C-12, D-12 Developing Edge)": {"lat": [33.695, 33.713], "lon": [72.930, 72.972]},
    "E-Sectors (E-7, E-8, E-9 Naval & Air Enclaves)": {"lat": [33.708, 33.726], "lon": [72.993, 73.035]},
    "F-6 / F-7 Sectors (Premium Mature Canopy)": {"lat": [33.712, 33.730], "lon": [73.035, 73.077]},
    "F-8 / F-10 / F-11 (Western Commercial Residential)": {"lat": [33.694, 33.712], "lon": [72.972, 73.035]},
    "G-6 / G-7 Sectors (High-Density Government Housing)": {"lat": [33.694, 33.712], "lon": [73.035, 73.077]},
    "G-8 / G-9 / G-10 / G-11 Sectors (Central Urban Residential)": {"lat": [33.676, 33.694], "lon": [72.972, 73.035]},
    "H-Sectors (H-8, H-9, H-11 Institutional Strip)": {"lat": [33.658, 33.676], "lon": [72.972, 73.077]},
    "I-8 Sector (High-Value Buffer Neighborhood)": {"lat": [33.676, 33.694], "lon": [73.056, 73.077]},
    "I-9 / I-10 Sectors (Industrial Engine Core)": {"lat": [33.658, 33.676], "lon": [73.035, 73.077]},
    "I-11 / I-12 (Western Transport Hub & Wholesale)": {"lat": [33.640, 33.658], "lon": [72.972, 73.035]},
    "Old Blue Area Central Axis (Jinnah Avenue Core)": {"lat": [33.705, 33.715], "lon": [73.040, 73.080]},
    "New Blue Area Westward Extension (F-9 / G-9 Edge)": {"lat": [33.685, 33.700], "lon": [73.010, 73.040]},
    "Fatima Jinnah Park (F-9 Mega Urban Forest Canopy)": {"lat": [33.695, 33.713], "lon": [72.993, 73.014]},
    "Shakarparian National Park & Lok Virsa Forest Complex": {"lat": [33.670, 33.690], "lon": [73.060, 73.090]},
    "Kachnar Park (I-8 Environmental Green Belt)": {"lat": [33.676, 33.682], "lon": [73.068, 73.075]},
    "Rawal Lake Basin Wetlands (Hydrological Cooler)": {"lat": [33.695, 33.720], "lon": [73.090, 73.130]},
    "Bani Gala Suburban Rim (Irregular Sprawl)": {"lat": [33.700, 33.725], "lon": [73.130, 73.170]},
    "Srinagar Highway Spine (Zero Point Gateway Corridor)": {"lat": [33.674, 33.680], "lon": [72.950, 73.060]},
    "Islamabad Expressway Spine (Faizabad Intersect Corridor)": {"lat": [33.620, 33.676], "lon": [73.075, 73.085]}
}

# 3. Sidebar Selection Panel
st.sidebar.header("🕹️ Regional Simulation Panel")
selected_zone = st.sidebar.selectbox("Target Geographical Sector", options=list(SECTORS.keys()))
intervention_pct = st.sidebar.slider("Target Zone Local Canopy Progress", min_value=0, max_value=100, value=0, step=20)

# 4. Spatial Processing Grid Setup & Simulation Loop
bounds = SECTORS[selected_zone]
df_simulated = df_pixels.copy()

# Isolate row index values falling inside our geofence polygon constraints
target_mask = (
    (df_simulated['Latitude'] >= bounds['lat'][0]) & (df_simulated['Latitude'] <= bounds['lat'][1]) &
    (df_simulated['Longitude'] >= bounds['lon'][0]) & (df_simulated['Longitude'] <= bounds['lon'][1])
)
target_idx = df_simulated[target_mask].index

fraction = intervention_pct / 100.0
max_ndvi = df_pixels['NDVI'].max()
min_ndbi = df_pixels['NDBI'].min()

# Apply targeted grid adaptations
if len(target_idx) > 0:
    df_simulated.loc[target_idx, 'NDVI'] += fraction * (max_ndvi - df_simulated.loc[target_idx, 'NDVI'])
    df_simulated.loc[target_idx, 'NDBI'] -= fraction * (df_simulated.loc[target_idx, 'NDBI'] - min_ndbi)

# MULTI-NEIGHBORHOOD THERMODYNAMIC SPILLOVER BLENDER
center_lat = np.mean(bounds['lat'])
center_lon = np.mean(bounds['lon'])

for name, geo in SECTORS.items():
    if name != selected_zone:
        sec_lat, sec_lon = np.mean(geo['lat']), np.mean(geo['lon'])
        distance = np.sqrt((center_lat - sec_lat)**2 + (center_lon - sec_lon)**2)

        # Check proximity boundary threshold (~2.5km)
        if distance < 0.06:
            spillover_weight = (1.0 - (distance / 0.06)) * fraction * 0.40
            neighbor_mask = (
                (df_simulated['Latitude'] >= geo['lat'][0]) & (df_simulated['Latitude'] <= geo['lat'][1]) &
                (df_simulated['Longitude'] >= geo['lon'][0]) & (df_simulated['Longitude'] <= geo['lon'][1])
            )
            neighbor_idx = df_simulated[neighbor_mask].index
            if len(neighbor_idx) > 0:
                df_simulated.loc[neighbor_idx, 'NDVI'] += spillover_weight * (max_ndvi - df_simulated.loc[neighbor_idx, 'NDVI'])
                df_simulated.loc[neighbor_idx, 'NDBI'] -= spillover_weight * (df_simulated.loc[neighbor_idx, 'NDBI'] - min_ndbi)

# Execute core matrix re-inference through Random Forest bypass validation
feature_order = ['NDVI', 'NDBI', 'Elevation']
df_simulated['Predicted_LST'] = rf_model.predict(df_simulated[feature_order].values)

# Isolate baseline metrics for the active targeted sector zone block
zone_baseline = df_pixels[target_mask]
zone_simulated = df_simulated[target_mask]

if len(zone_baseline) > 0:
    pred_baseline_temp = zone_baseline['LST'].mean()
    pred_current_temp = zone_simulated['Predicted_LST'].mean()
else:
    # Safe fallback if a zone has thin pixel density samples
    pred_baseline_temp = df_pixels['LST'].mean()
    pred_current_temp = df_simulated['Predicted_LST'].mean()

live_cooling_delta = pred_baseline_temp - pred_current_temp

# 5. Scorecard Matrix Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Selected Zone Baseline Avg", value=f"{pred_baseline_temp:.2f} °C")
with col2:
    st.metric(label="Simulated Zone Status Avg", value=f"{pred_current_temp:.2f} °C",
              delta=f"-{live_cooling_delta:.2f} °C" if live_cooling_delta > 0.01 else None, delta_color="inverse")
with col3:
    st.metric(label="City-Wide General Effect", value=f"-{df_pixels['LST'].mean() - df_simulated['Predicted_LST'].mean():.3f} °C Total Delta")

st.write("---")

# 6. Layout Presentation Columns
left_chart_column, right_map_column = st.columns([2, 3])

with left_chart_column:
    st.subheader("📊 Sector Phase Progression Trend")

    pct_steps = [0, 20, 40, 60, 80, 100]
    temps_trend = []

    # Process graph line coordinate arrays safely using deep grid iterations
    for p in pct_steps:
        df_step = df_pixels.copy()
        f = p / 100.0
        if len(target_idx) > 0:
            df_step.loc[target_idx, 'NDVI'] += f * (max_ndvi - df_step.loc[target_idx, 'NDVI'])
            df_step.loc[target_idx, 'NDBI'] -= f * (df_step.loc[target_idx, 'NDBI'] - min_ndbi)

        step_preds = rf_model.predict(df_step[feature_order].values)
        if len(target_idx) > 0:
            temps_trend.append(step_preds[target_idx].mean())
        else:
            temps_trend.append(step_preds.mean())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(pct_steps, temps_trend, color='#e74c3c', marker='o', linewidth=2.5, label="Zone Trajectory")
    ax.axvline(x=intervention_pct, color='#3498db', linestyle='--', linewidth=2, label="Active Phase")
    ax.set_xlabel("Greening Progress (%)", fontweight='bold')
    ax.set_ylabel("Predicted Zone Surface Temp (°C)", fontweight='bold')
    ax.legend()
    st.pyplot(fig)

with right_map_column:
    st.subheader("🗺️ Islamabad Satellite Heatmap Layer")
    st.write("Interactions fully stabilized. Zoom and pan freely across the microclimates:")

    # Calculate dataset temperature bounds for our visual normalization weights
    min_lst = df_simulated['Predicted_LST'].min()
    max_lst = df_simulated['Predicted_LST'].max()

    # Build Map Instance centered precisely on the currently targeted sector zone block
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        min_zoom=10, max_zoom=16,
        tiles="CartoDB dark_matter"
    )

    # Format data list while normalizing weights
    heat_data = [
        [row['Latitude'], row['Longitude'], (row['Predicted_LST'] - min_lst) / (max_lst - min_lst)]
        for _, row in df_simulated.iterrows()
    ]

    HeatMap(data=heat_data, radius=22, blur=15, max_zoom=13, min_opacity=0.30).add_to(m)

    # Floating HTML/CSS Color Legend Component Canvas Overlay
    legend_html = f"""
     <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: 90px; background-color: rgba(30, 30, 30, 0.85); backdrop-filter: blur(5px); z-index:9999; font-size:12px; color: white; padding: 10px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.5); font-family: sans-serif; border: 1px solid rgba(255,255,255,0.1);">
     <b style="color: #f3f4f6;">Simulated Surface Temp (LST)</b><br>
     <div style="background: linear-gradient(to right, blue, cyan, green, yellow, orange, red); height: 12px; width: 100%; border-radius: 3px; margin-top: 8px; margin-bottom: 5px;"></div>
     <span style="float: left;">Cool ({min_lst:.1f}°C)</span><br><span style="float: right;">Hot ({max_lst:.1f}°C)</span>
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Frame active policy intervention zone with bounding box layout lines
    folium.Rectangle(
        bounds=[[bounds['lat'][0], bounds['lon'][0]], [bounds['lat'][1], bounds['lon'][1]]],
        color="#2ecc71", weight=3, fill=True, fill_color="#2ecc71", fill_opacity=0.10,
        popup=folium.Popup(f"<b>Active Target Zone:</b><br>{selected_zone}", max_width=200)
    ).add_to(m)

    # Render via static element frame to kill infinite server refreshing ping-pong matches
    folium_static(m, height=450, width=680)