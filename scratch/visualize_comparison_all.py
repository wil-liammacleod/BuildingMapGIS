import pydeck as pdk
import geopandas as gpd
import os
import pandas as pd
import json

def create_comparison_viz(building_id):
    raw_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    
    round_scenarios = {
        'Raw Input': {
            'file': raw_path, 
            'color': [255, 0, 0, 100], 
            'offset_idx': -2, 
            'label': 'Raw Input (Overlapping, Unhealed)'
        },
        'Method D (No Rounding)': {
            'file': f"artifacts/building_{building_id}_method_d_res_0.25_round_none.geojson", 
            'color': [255, 150, 0, 200], 
            'offset_idx': -0.5, 
            'label': 'Method D (No Rounding)'
        },
        'Method D (1m Rounding)': {
            'file': f"artifacts/building_{building_id}_method_d_res_0.25_round_1.0m.geojson", 
            'color': [255, 215, 0, 200], 
            'offset_idx': 0.5, 
            'label': 'Method D (1m Rounding - HVAC Merged)'
        },
        'Method D (2m Rounding)': {
            'file': f"artifacts/building_{building_id}_method_d_res_0.25_round_2.0m.geojson", 
            'color': [255, 99, 71, 200], 
            'offset_idx': 1.5, 
            'label': 'Method D (2m Rounding - Coarse Profile)'
        },
        'Method D (3m Rounding)': {
            'file': f"artifacts/building_{building_id}_method_d_res_0.25_round_3.0m.geojson", 
            'color': [218, 112, 214, 200], 
            'offset_idx': 2.5, 
            'label': 'Method D (3m Rounding - Major Levels)'
        }
    }
    
    # Load raw data to find center
    gdf_raw_full = gpd.read_file(raw_path)
    gdf_raw = gdf_raw_full[gdf_raw_full['BUILDING'] == building_id].to_crs("EPSG:4326")
    
    if gdf_raw.empty:
        print(f"Building {building_id} not found in raw data.")
        return
        
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
    spacing = 0.0008  # spacing factor for side-by-side shift
    
    # For each scenario, load the GeoJSON, apply the offset, and add to layers
    for name, config in round_scenarios.items():
        file_path = config['file']
        if not os.path.exists(file_path):
            print(f"Skipping {name}: file not found at {file_path}")
            continue
            
        gdf = gpd.read_file(file_path)
        
        # Filter for building_id if it's the raw file
        if name == 'Raw Input':
            gdf = gdf[gdf['BUILDING'] == building_id].copy()
            
        if gdf.empty:
            continue
            
        # Ensure CRS is 4326
        gdf = gdf.to_crs("EPSG:4326")
        
        # Extract total stats from first row if available, else compute
        total_sa = gdf['clean_surface_area'].iloc[0] if 'clean_surface_area' in gdf.columns else 0.0
        total_vol = gdf['clean_volume_total'].iloc[0] if 'clean_volume_total' in gdf.columns else 0.0
        total_sqft = gdf['total_internal_sqft'].iloc[0] if 'total_internal_sqft' in gdf.columns else 0.0
        max_floors = gdf['max_floors'].iloc[0] if 'max_floors' in gdf.columns else 0
        bldg_use = gdf['bldg_use'].iloc[0] if 'bldg_use' in gdf.columns else 'N/A'
        
        # Calculate vertices
        total_verts = 0
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'Polygon':
                total_verts += len(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                for p in geom.geoms:
                    total_verts += len(p.exterior.coords)
                    
        # Apply properties to rows for tooltips
        gdf['method_label'] = config['label']
        gdf['bldg_total_sa'] = f"{total_sa:,.1f}" if total_sa > 0 else "N/A"
        gdf['bldg_total_vol'] = f"{total_vol:,.1f}" if total_vol > 0 else "N/A"
        gdf['bldg_total_sqft'] = f"{total_sqft:,.1f}" if total_sqft > 0 else "N/A"
        gdf['bldg_max_floors'] = int(max_floors) if max_floors > 0 else "N/A"
        gdf['bldg_use'] = bldg_use
        gdf['bldg_shapes_count'] = len(gdf)
        gdf['bldg_vertices_count'] = total_verts
        
        # Shift horizontally
        gdf.geometry = gdf.geometry.translate(xoff=config['offset_idx'] * spacing)
        
        # Create pydeck layer
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            gdf,
            opacity=0.4 if name == 'Raw Input' else 0.8,
            extruded=True,
            get_elevation="height_p90",
            get_fill_color=config['color'],
            get_line_color=[255, 255, 255],
            pickable=True,
        ))
        
        print(f"Added layer: {name} (Shapes: {len(gdf)}, Vertices: {total_verts}, Internal: {total_sqft:,.1f} sqft, Floors: {max_floors})")

    # Create Deck
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "html": """
                <b>Method/Config:</b> {method_label}<br/>
                <hr/>
                <b>Segment Height:</b> {height_p90}m<br/>
                <b>Segment Area:</b> {clean_area}m²<br/>
                <b>Segment Floors:</b> {num_floors}<br/>
                <b>Segment Int Area:</b> {internal_area_sqft} sq ft<br/>
                <hr/>
                <b>BUILDING TOTALS:</b><br/>
                <b>Estimated Use:</b> {bldg_use}<br/>
                <b>Max Floors:</b> {bldg_max_floors}<br/>
                <b>Total Floor Area:</b> {bldg_total_sqft} sq ft<br/>
                <b>Sealed Surface Area:</b> {bldg_total_sa} m²<br/>
                <b>Total Volume:</b> {bldg_total_vol} m³<br/>
                <b>Shapes:</b> {bldg_shapes_count} | <b>Vertices:</b> {bldg_vertices_count}
            """,
            "style": {"color": "white", "backgroundColor": "#222", "fontSize": "12px"}
        },
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    
    output_html = "artifacts/all_methods_comparison.html"
    r.to_html(output_html)
    print(f"Comparison visualization saved to: {output_html}")
    print("Red (x-2) = Raw | Orange (x-0.5) = Method D (No Round) | Gold (x+0.5) = 1m Round | Coral (x+1.5) = 2m Round | Magenta (x+2.5) = 3m Round")

if __name__ == "__main__":
    create_comparison_viz(building_id=132)
