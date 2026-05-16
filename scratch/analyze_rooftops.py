import geopandas as gpd
import pandas as pd
import os

def analyze_rooftops(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    gdf = gpd.read_file(file_path)
    
    total_segments = len(gdf)
    num_buildings = gdf['BUILDING'].nunique()
    
    print(f"\n--- Rooftop Segmentation Analysis ---")
    print(f"Total Planar Segments: {total_segments}")
    print(f"Total Unique Buildings: {num_buildings}")
    
    # Count segments per building
    counts = gdf.groupby('BUILDING').size().reset_index(name='object_count')
    
    # Get some statistics on object counts
    print(f"\nObjects per Building Statistics:")
    print(counts['object_count'].describe())
    
    # Top 10 buildings by object count
    print(f"\nTop 10 Buildings by Number of Objects:")
    print(counts.sort_values(by='object_count', ascending=False).head(10).to_string(index=False))
    
    # Save the full list to a CSV for reference if needed
    output_csv = "artifacts/building_object_counts.csv"
    os.makedirs("artifacts", exist_ok=True)
    counts.to_csv(output_csv, index=False)
    print(f"\nFull building object counts saved to: {output_csv}")

if __name__ == "__main__":
    rooftops_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    analyze_rooftops(rooftops_path)
