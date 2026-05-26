"""
Isb-Thermal-Pulse: Interactive Production Web App
Uses Streamlit to build a graphic user interface dashboard, allowing users to
dynamically interact with our trained Random Forest model, Folium maps, and analytic plots.
"""

import os
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from config import settings

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Isb-Thermal-Pulse Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Isb-Thermal-Pulse: Urban Forestry ML Simulator")
st.markdown("### Dynamic Microclimate Intervention Matrix for Islamabad (2-Kanal Plots)")
st.write("---")


# 2. Load the Pre-Trained Model Brain Once
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
    st.error(
        "🚨 Missing System Artifacts. Please run 'python main.py' in your terminal first to initialize the dataset and train the model weights!")
    st.stop()

# 3. Create Sidebar Panel for Interactive Controls
st.sidebar.header("🕹️ Simulation Intervention Controls")
st.sidebar.write("Configure your urban greening parameters below:")

# Select Elevation Context Zone
elevation_profile = st.sidebar.selectbox(
    "Target Sector Elevation Baseline",
    options=["Valley Floor (Flat Urban Sectors ~500m)", "Margalla Foothills (~650m)"]
)

# Set elevation numerical threshold ranges based on user selection
if "Valley Floor" in elevation_profile:
    sub_df = df_pixels[(df_pixels['Elevation'] >= 490) & (df_pixels['Elevation'] <= 530)]
else:
    sub_df = df_pixels[(df_pixels['Elevation'] >= 600) & (df_pixels['Elevation'] <= 700)]

# Extract real-world data bounds
hotspot = sub_df.sort_values(by='NDBI', ascending=False).iloc[0]
green_target = sub_df.sort_values(by='NDVI', ascending=False).iloc[0]

# Interactive Greening Slider Control
intervention_pct = st.sidebar.slider(
    "Reforestation Canopy Density Expansion Phase",
    min_value=0,
    max_value=100,
    value=40,
    step=20,
    help="0% represents absolute concrete pavement baseline. 100% represents a mature interlocking urban park canopy."
)

# 4. Perform Live Machine Learning Inference
fraction = intervention_pct / 100.0
simulated_ndvi = hotspot['NDVI'] + fraction * (green_target['NDVI'] - hotspot['NDVI'])
simulated_ndbi = hotspot['NDBI'] + fraction * (green_target['NDBI'] - hotspot['NDBI'])

feature_order = ['NDVI', 'NDBI', 'Elevation']
baseline_row = pd.DataFrame([{'NDVI': hotspot['NDVI'], 'NDBI': hotspot['NDBI'], 'Elevation': hotspot['Elevation']}])[
    feature_order]
simulated_row = pd.DataFrame([{'NDVI': simulated_ndvi, 'NDBI': simulated_ndbi, 'Elevation': hotspot['Elevation']}])[
    feature_order]

pred_baseline_temp = rf_model.predict(baseline_row)[0]
pred_current_temp = rf_model.predict(simulated_row)[0]
live_cooling_delta = pred_baseline_temp - pred_current_temp

# 5. Display KPI Dashboard Scorecards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Baseline Pavement Temperature", value=f"{pred_baseline_temp:.2f} °C")
with col2:
    st.metric(label="Simulated Phase Temperature", value=f"{pred_current_temp:.2f} °C",
              delta=f"-{live_cooling_delta:.2f} °C" if live_cooling_delta > 0 else f"+{abs(live_cooling_delta):.2f} °C",
              delta_color="inverse")
with col3:
    st.metric(label="Local Shading Efficiency Status",
              value="Critical Heat Island" if live_cooling_delta < 1.0 else "Microclimate Shield Active")

st.write("---")

# 6. Main Dashboard Layout Columns
left_chart_column, right_map_column = st.columns([1, 1])

with left_chart_column:
    st.subheader("📊 Dynamic Microclimate Performance Matrix")

    # Generate live loop data array to show phase line trend charts
    pct_steps = [0, 20, 40, 60, 80, 100]
    temps_trend = []
    for p in pct_steps:
        f = p / 100.0
        s_ndvi = hotspot['NDVI'] + f * (green_target['NDVI'] - hotspot['NDVI'])
        s_ndbi = hotspot['NDBI'] + f * (green_target['NDBI'] - hotspot['NDBI'])
        s_row = pd.DataFrame([{'NDVI': s_ndvi, 'NDBI': s_ndbi, 'Elevation': hotspot['Elevation']}])[feature_order]
        temps_trend.append(rf_model.predict(s_row)[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pct_steps, temps_trend, color='#e74c3c', marker='o', linewidth=2.5, label="Temperature Path")
    ax.axvline(x=intervention_pct, color='#3498db', linestyle='--', linewidth=2, label="Current Selection")
    ax.set_xlabel("Greening Implementation Progress (%)", fontweight='bold')
    ax.set_ylabel("Predicted Surface Temp (°C)", fontweight='bold')
    ax.legend()
    st.pyplot(fig)

with right_map_column:
    st.subheader("📍 Real-World Coordinate Intervention View")

    # Extract the true coordinates calculated by Option 1!
    target_lat = hotspot['Latitude']
    target_lon = hotspot['Longitude']

    # Construct an interactive live Folium layer
    m = folium.Map(location=[target_lat, target_lon], zoom_start=14, tiles="CartoDB positron")

    popup_card = f"""
    <div style='font-family: sans-serif; width:180px;'>
        <h5>Target Plot View</h5>
        <b>Current Green Cover:</b> {intervention_pct}%<br>
        <b>Live Temp:</b> {pred_current_temp:.2f}°C
    </div>
    """

    folium.Marker(
        location=[target_lat, target_lon],
        popup=folium.Popup(popup_card, max_width=250),
        icon=folium.Icon(color="red" if live_cooling_delta < 1.0 else "green",
                         icon="leaf" if live_cooling_delta >= 1.0 else "fire")
    ).add_to(m)

    # Render map component within Streamlit UI cleanly
    st_folium(m, height=350, use_container_width=True)