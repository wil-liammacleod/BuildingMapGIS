import pydeck as pdk
import geopandas as gpd
import os
import pandas as pd
import json

def create_comparison_viz(building_id):
    raw_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    method_a_path = f"artifacts/building_{building_id}_method_a.geojson"
    method_b_path = f"artifacts/building_{building_id}_method_b.geojson"
    method_c_path = f"artifacts/building_{building_id}_method_c.json"
    
    # Load Method C results
    method_c_stats = {}
    if os.path.exists(method_c_path):
        with open(method_c_path, 'r') as f:
            method_c_stats = json.load(f)
    
    # Load data
    gdf_raw = gpd.read_file(raw_path)
    gdf_raw = gdf_raw[gdf_raw['BUILDING'] == building_id].to_crs("EPSG:4326")
    
    # Calculate Raw totals
    # We need to re-project to 2958 for accurate area sum
    raw_2958 = gdf_raw.to_crs("EPSG:2958")
    raw_total_area = raw_2958['AREA'].sum()
    
    # Load method results
    methods = {}
    if os.path.exists(method_a_path):
        methods['Method A (TopoJSON)'] = gpd.read_file(method_a_path).to_crs("EPSG:4326")
    if os.path.exists(method_b_path):
        methods['Method B (Trimesh)'] = gpd.read_file(method_b_path).to_crs("EPSG:4326")
    
    # Method C GeoJSON
    method_c_geojson = f"artifacts/building_{building_id}_method_c.geojson"
    if os.path.exists(method_c_geojson):
        methods['Method C (Voxel)'] = gpd.read_file(method_c_geojson).to_crs("EPSG:4326")

    # View State
    center_lat = gdf_raw.geometry.centroid.y.mean()
    center_lon = gdf_raw.geometry.centroid.x.mean()
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=18.5,
        pitch=45,
        bearing=0
    )

    layers = []
    spacing = 0.0008
    
    # 1. Raw Layer (Red)
    raw_offset = gdf_raw.copy()
    raw_offset['method_label'] = 'Raw Input (Overlapping)'
    raw_offset['bldg_total_area'] = f"{raw_total_area:,.2f}"
    raw_offset['bldg_total_vol'] = "N/A"
    raw_offset['method_c_ref'] = f"{method_c_stats.get('total_surface_area', 'N/A')} m²"
    raw_offset.geometry = raw_offset.geometry.translate(xoff=-spacing * 1.5)
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        raw_offset,
        opacity=0.4,
        extruded=True,
        get_elevation="height_p90",
        get_fill_color=[255, 0, 0, 100],
        get_line_color=[255, 255, 255],
        pickable=True,
    ))

    # 2. Method A Layer (Green)
    if 'Method A (TopoJSON)' in methods:
        a_gdf = methods['Method A (TopoJSON)'].copy()
        a_gdf['method_label'] = 'Method A (TopoJSON)'
        total_sa = a_gdf['clean_surface_area'].sum() if 'clean_surface_area' in a_gdf.columns else 0
        total_vol = a_gdf['clean_volume'].sum()
        a_gdf['bldg_total_area'] = f"{total_sa:,.2f}"
        a_gdf['bldg_total_vol'] = f"{total_vol:,.2f}"
        a_gdf['method_c_ref'] = f"{method_c_stats.get('total_surface_area', 'N/A')} m²"
        a_gdf.geometry = a_gdf.geometry.translate(xoff=-spacing * 0.5)
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            a_gdf,
            opacity=0.8,
            extruded=True,
            get_elevation="height_p90",
            get_fill_color=[0, 200, 100, 200],
            get_line_color=[255, 255, 255],
            pickable=True,
        ))

    # 3. Method B Layer (Blue)
    if 'Method B (Trimesh)' in methods:
        b_offset = methods['Method B (Trimesh)'].copy()
        b_offset['method_label'] = 'Method B (Trimesh)'
        total_vol = b_offset['clean_volume'].sum()
        b_offset['bldg_total_area'] = "See console for Mesh SA"
        b_offset['bldg_total_vol'] = f"{total_vol:,.2f}"
        b_offset['method_c_ref'] = f"{method_c_stats.get('total_surface_area', 'N/A')} m²"
        b_offset.geometry = b_offset.geometry.translate(xoff=spacing * 0.5)
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            b_offset,
            opacity=0.8,
            extruded=True,
            get_elevation="height_p90",
            get_fill_color=[0, 100, 255, 200],
            get_line_color=[255, 255, 255],
            pickable=True,
        ))

    # 4. Method C Layer (Purple)
    if 'Method C (Voxel)' in methods:
        c_offset = methods['Method C (Voxel)'].copy()
        c_offset['method_label'] = 'Method C (Voxelized)'
        c_offset['bldg_total_area'] = f"{method_c_stats.get('total_surface_area', 'N/A'):,.2f}"
        c_offset['bldg_total_vol'] = f"{method_c_stats.get('total_volume', 'N/A'):,.2f}"
        c_offset['method_c_ref'] = "SELF"
        c_offset.geometry = c_offset.geometry.translate(xoff=spacing * 1.5)
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            c_offset,
            opacity=0.9,
            extruded=True,
            get_elevation="height_p90",
            get_fill_color=[150, 0, 255, 200],
            get_line_color=[200, 200, 200],
            pickable=True,
        ))

    # Create Deck
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "html": """
                <b>Method:</b> {method_label}<br/>
                <hr/>
                <b>Segment Height:</b> {height_p90}m<br/>
                <b>Segment Area:</b> {clean_area}m²<br/>
                <hr/>
                <b>BUILDING TOTALS:</b><br/>
                <b>Total SA (this method):</b> {bldg_total_area} m²<br/>
                <b>Total Vol (this method):</b> {bldg_total_vol} m³<br/>
                <hr/>
                <b>Method C (Voxel Reference) SA:</b> {method_c_ref}
            """,
            "style": {"color": "white", "backgroundColor": "#222", "fontSize": "12px"}
        },
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    
    output_html = "artifacts/all_methods_comparison.html"
    r.to_html(output_html)
    print(f"Comparison visualization saved to: {output_html}")
    print("Red (Left) = Raw | Green (Center) = Method A | Blue (Right) = Method B")

if __name__ == "__main__":
    create_comparison_viz(building_id=132)
