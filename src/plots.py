"""
Isb-Thermal-Pulse: Analytics Plotting Layer
Uses Matplotlib to generate static scientific charts tracking
microclimate cooling performance across project simulation phases.
"""

import os
import matplotlib.pyplot as plt
from config import settings

class ThermalPlotter:
    def __init__(self):
        # Set a clean, professional aesthetic style
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    def generate_trend_chart(self, phases, temps, coolings):
        """Generates a dual-axis line chart tracking temperature vs cooling phases"""
        print("\nCompiling scientific trend chart via Matplotlib...")

        fig, ax1 = plt.subplots(figsize=(10, 5.5))

        # Primary Axis: Absolute Surface Temperature (Red Line)
        color_temp = '#e74c3c'
        ax1.set_xlabel('Simulation Greening Phases', fontweight='bold', labelpad=10)
        ax1.set_ylabel('Predicted Surface Temp (°C)', color=color_temp, fontweight='bold')
        ax1.plot(phases, temps, color=color_temp, marker='o', linewidth=2.5, markersize=8, label='Surface Temp')
        ax1.tick_params(axis='y', labelcolor=color_temp)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Secondary Axis: Cumulative Cooling Delta (Green Dashed Line)
        ax2 = ax1.twinx()
        color_cool = '#2ecc71'
        ax2.set_ylabel('Cumulative Cooling Delta (ΔT °C)', color=color_cool, fontweight='bold')
        ax2.plot(phases, coolings, color=color_cool, marker='s', linestyle='--', linewidth=2, markersize=8, label='Cooling Delta')
        ax2.tick_params(axis='y', labelcolor=color_cool)

        # Labels, Titles, and Layout styling
        plt.title('Islamabad Microclimate Response Matrix\n(2-Kanal Plot Simulation Performance)',
                  fontsize=13, fontweight='bold', pad=15)

        fig.tight_layout()

        # Save chart cleanly to your processed data folder
        output_chart_path = os.path.join(settings.PROCESSED_DATA_DIR, 'simulation_trend_chart.png')
        plt.savefig(output_chart_path, dpi=300)
        plt.close() # Closes the canvas to clear system memory

        print(f"SUCCESS: Scientific trend chart compiled and saved to: {output_chart_path}")