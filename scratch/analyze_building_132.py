import geopandas as gpd
import pandas as pd
import os

def analyze_building_132(file_path):
    gdf = gpd.read_file(file_path)
    b132 = gdf[gdf['BUILDING'] == 132].copy()
    
    if b132.empty:
        print("Building 132 not found.")
        return

    print(f"Building 132 - Detailed Segment Analysis")
    print(f"Total Segments: {len(b132)}")
    print("-" * 50)
    
    # Calculate vertex count for each geometry
    b132['vertices'] = b132.geometry.apply(lambda g: len(g.exterior.coords) if g.geom_type == 'Polygon' else sum(len(p.exterior.coords) for p in g.geoms))
    
    # Summary of segments
    summary = b132[['height_p90', 'AREA', 'SLOPE', 'ASPECT', 'vertices']].copy()
    summary.columns = ['Height (m)', 'Area (m²)', 'Slope (°)', 'Aspect (°)', 'Vertices']
    
    print(summary.to_string(index=False))
    
    print("\n--- Aggregate Statistics for Building 132 ---")
    print(f"Total Area: {b132['AREA'].sum():.2f} m²")
    print(f"Height Range: {b132['height_p90'].min()}m - {b132['height_p90'].max()}m")
    print(f"Slope Range: {b132['SLOPE'].min():.1f}° - {b132['SLOPE'].max():.1f}°")
    
    # Diversity of aspects (to see if it's a multi-gabled roof)
    aspects = b132['ASPECT'].round(-1) # Group by 10 degrees
    unique_aspect_bins = aspects.nunique()
    print(f"Unique Aspect Orientations (10° bins): {unique_aspect_bins}")

if __name__ == "__main__":
    rooftops_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    analyze_building_132(rooftops_path)
