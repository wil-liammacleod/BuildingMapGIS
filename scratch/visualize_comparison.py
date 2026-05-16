import pydeck as pdk
import geopandas as gpd
import os

def create_comparison_viz(input_path, output_path, building_id):
    # Load data
    gdf_in = gpd.read_file(input_path)
    gdf_in = gdf_in[gdf_in['BUILDING'] == building_id].to_crs("EPSG:4326")
    
    gdf_out = gpd.read_file(output_path).to_crs("EPSG:4326")
    
    # Calculate map center
    center_lat = gdf_in.geometry.centroid.y.mean()
    center_lon = gdf_in.geometry.centroid.x.mean()
    
    # Layer 1: Raw (Red)
    # Offset the raw data slightly to the left for side-by-side comparison
    gdf_in_offset = gdf_in.copy()
    gdf_in_offset.geometry = gdf_in_offset.geometry.translate(xoff=-0.001)
    
    layer_raw = pdk.Layer(
        "GeoJsonLayer",
        gdf_in_offset,
        opacity=0.5,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="height_p90",
        get_fill_color=[255, 0, 0, 150],
        get_line_color=[255, 255, 255],
        pickable=True
    )
    
    # Layer 2: Cleaned (Green)
    # Offset to the right
    gdf_out_offset = gdf_out.copy()
    gdf_out_offset.geometry = gdf_out_offset.geometry.translate(xoff=0.001)
    
    layer_clean = pdk.Layer(
        "GeoJsonLayer",
        gdf_out_offset,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="height_p90",
        get_fill_color=[0, 200, 100, 200],
        get_line_color=[255, 255, 255],
        pickable=True
    )
    
    # View State
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=17,
        pitch=45,
        bearing=0
    )
    
    # Create Deck
    r = pdk.Deck(
        layers=[layer_raw, layer_clean],
        initial_view_state=view_state,
        tooltip={"text": "Height: {height_p90}m\nArea: {AREA}m²"},
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    
    # Save to HTML
    output_html = "artifacts/comparison_viz.html"
    r.to_html(output_html)
    print(f"Comparison visualization saved to: {output_html}")
    print("Red (Left) = Raw Input | Green (Right) = Cleaned Output")

if __name__ == "__main__":
    input_file = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    output_file = "artifacts/building_132_block_model.geojson"
    
    if os.path.exists(output_file):
        create_comparison_viz(input_file, output_file, building_id=132)
    else:
        print("Output file not found. Run the cleaning script first.")
