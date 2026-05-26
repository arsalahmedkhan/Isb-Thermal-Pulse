"""
Isb-Thermal-Pulse: Visualization Layer (Fixed Local Intervention)
Generates an interactive HTML map focusing entirely on the target intervention plot,
displaying its phase-by-phase local cooling transformation.
"""

import os
import folium
from config import settings

class ThermalVisualizer:
    def __init__(self):
        self.islamabad_center = [33.6844, 73.0479]

    def generate_intervention_map(self, hotspot_coords, baseline_temp, mitigated_temp, cooling_delta):
        """
        Creates an interactive map focused on ONE single location, showing how
        its properties change locally through urban forestry.
        """
        print("\nCompiling localized spatial intervention layer...")

        # Initialize the base map
        m = folium.Map(
            location=hotspot_coords,  # Center the map directly on the hot spot!
            zoom_start=14,            # Zoom closer so we can see the local sector
            tiles="CartoDB positron"
        )

        # Build an HTML-formatted timeline table to put inside the single marker's popup
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #e74c3c;">2-Kanal Urban Intervention</h4>
            <p><b>Location Coordinates:</b> {hotspot_coords[0]:.4f}, {hotspot_coords[1]:.4f}</p>
            <hr style="border: 0; border-top: 1px solid #ccc; margin: 10px 0;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f2f2f2;">
                    <th style="text-align: left; padding: 4px;">Phase</th>
                    <th style="text-align: right; padding: 4px;">Predicted Temp</th>
                </tr>
                <tr>
                    <td style="padding: 4px; color: #e74c3c;"><b>Baseline (0%)</b></td>
                    <td style="text-align: right; padding: 4px; color: #e74c3c;"><b>{baseline_temp:.2f}°C</b></td>
                </tr>
                <tr>
                    <td style="padding: 4px; color: #2ecc71;"><b>Mitigated (100%)</b></td>
                    <td style="text-align: right; padding: 4px; color: #2ecc71;"><b>{mitigated_temp:.2f}°C</b></td>
                </tr>
            </table>
            <hr style="border: 0; border-top: 1px solid #ccc; margin: 10px 0;">
            <p style="margin: 0; font-weight: bold; color: #2980b9;">
                Local Shading Power (ΔT): -{cooling_delta:.2f}°C
            </p>
        </div>
        """

        # Add ONE single marker that changes properties
        folium.Marker(
            location=hotspot_coords,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip="Click to view localized planting intervention timeline",
            icon=folium.Icon(color="darkred", icon="cloud")
        ).add_to(m)

        # Save the updated local map
        output_path = os.path.join(os.path.dirname(settings.RAW_DATA_DIR), "islamabad_thermal_map.html")
        m.save(output_path)

        print(f"SUCCESS: Localized intervention map updated at: {output_path}")