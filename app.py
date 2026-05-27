"""
Isb-Thermal-Pulse: Advanced Map Studio
Features full city-wide thermal heatmap layer mapping using your safe,
stable 5,000-point empirical database framework.
"""

import os
import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from config import settings

# 1. Page Configuration
st.set_page_config(
    page_title="Isb-Thermal-Pulse Map Studio",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Isb-Thermal-Pulse: Spatial Analytics Dashboard")
st.markdown("### Regional Satellite Thermal Gradients & Localized Intervention Matrix")
st.write("---")

# 2. Optimized Data Loading
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
    st.error("🚨 Configuration Error: Run 'python main.py' to verify database compilation!")
    st.stop()

# 3. Sidebar Simulation Panel
st.sidebar.header("🕹️ Regional Simulation Panel")

elevation_profile = st.sidebar.selectbox(
    "Target Topographic Zone",
    options=["Valley Floor (Flat Urban Sectors ~500m)", "Margalla Foothills (~650m)"]
)

if "Valley Floor" in elevation_profile:
    sub_df = df_pixels[(df_pixels['Elevation'] >= 490) & (df_pixels['Elevation'] <= 530)]
else:
    sub_df = df_pixels[(df_pixels['Elevation'] >= 600) & (df_pixels['Elevation'] <= 700)]

hotspot = sub_df.sort_values(by='NDBI', ascending=False).iloc[0]
green_target = sub_df.sort_values(by='NDVI', ascending=False).iloc[0]

intervention_pct = st.sidebar.slider(
    "Target Plot Local Canopy Progress",
    min_value=0, max_value=100, value=0, step=20
)

# 4. Live ML Inference
fraction = intervention_pct / 100.0
simulated_ndvi = hotspot['NDVI'] + fraction * (green_target['NDVI'] - hotspot['NDVI'])
simulated_ndbi = hotspot['NDBI'] + fraction * (green_target['NDBI'] - hotspot['NDBI'])

feature_order = ['NDVI', 'NDBI', 'Elevation']
baseline_row = pd.DataFrame([{'NDVI': hotspot['NDVI'], 'NDBI': hotspot['NDBI'], 'Elevation': hotspot['Elevation']}])[feature_order]
simulated_row = pd.DataFrame([{'NDVI': simulated_ndvi, 'NDBI': simulated_ndbi, 'Elevation': hotspot['Elevation']}])[feature_order]

pred_baseline_temp = rf_model.predict(baseline_row)[0]
pred_current_temp = rf_model.predict(simulated_row)[0]
live_cooling_delta = pred_baseline_temp - pred_current_temp

# 5. Scorecard Summaries
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Target Plot Baseline", value=f"{pred_baseline_temp:.2f} °C")
with col2:
    st.metric(label="Simulated Phase Status", value=f"{pred_current_temp:.2f} °C",
              delta=f"-{live_cooling_delta:.2f} °C" if live_cooling_delta > 0 else f"+{abs(live_cooling_delta):.2f} °C", delta_color="inverse")
with col3:
    st.metric(label="Thermal Management Mode", value="Critical Alert" if live_cooling_delta < 2.0 else "Microclimate Shield Active")

st.write("---")

# 6. Presentation Columns
left_chart_column, right_map_column = st.columns([2, 3])

with left_chart_column:
    st.subheader("📊 Model Phase Progression Trend")

    pct_steps = [0, 20, 40, 60, 80, 100]
    temps_trend = []
    for p in pct_steps:
        f = p / 100.0
        s_ndvi = hotspot['NDVI'] + f * (green_target['NDVI'] - hotspot['NDVI'])
        s_ndbi = hotspot['NDBI'] + f * (green_target['NDBI'] - hotspot['NDBI'])
        s_row = pd.DataFrame([{'NDVI': s_ndvi, 'NDBI': s_ndbi, 'Elevation': hotspot['Elevation']}])[feature_order]
        temps_trend.append(rf_model.predict(s_row)[0])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(pct_steps, temps_trend, color='#e74c3c', marker='o', linewidth=2.5, label="Microclimate Vector")
    ax.axvline(x=intervention_pct, color='#3498db', linestyle='--', linewidth=2, label="Active Phase")
    ax.set_xlabel("Greening Progress (%)", fontweight='bold')
    ax.set_ylabel("Predicted Surface Temp (°C)", fontweight='bold')
    ax.legend()
    st.pyplot(fig)

with right_map_column:
    st.subheader("🗺️ Islamabad Satellite Heatmap Layer")

    target_lat = hotspot['Latitude']
    target_lon = hotspot['Longitude']

    # Render with premium slate-black basemap tiles
    m = folium.Map(location=[33.6844, 73.0479], zoom_start=12, tiles="CartoDB dark_matter")

    # Extract coordinates and satellite LST for the heatmap overlay
    heat_data = df_pixels[['Latitude', 'Longitude', 'LST']].values.tolist()

    # Inject safe, smoothly-contoured heatmap density layer
    HeatMap(
        data=heat_data,
        radius=15,
        blur=12,
        max_zoom=13,
        min_opacity=0.3
    ).add_to(m)

    popup_card = f"""
    <div style='font-family: sans-serif; width:180px; color:#2c3e50;'>
        <h5>Active Simulation Plot</h5>
        <b>Current Green Cover:</b> {intervention_pct}%<br>
        <b>Live Target Temp:</b> {pred_current_temp:.2f}°C
    </div>
    """
    folium.Marker(
        location=[target_lat, target_lon],
        popup=folium.Popup(popup_card, max_width=250),
        icon=folium.Icon(color="red" if live_cooling_delta < 2.0 else "green", icon="info-sign")
    ).add_to(m)

    st_folium(m, height=450, use_container_width=True)