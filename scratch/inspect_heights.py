import geopandas as gpd
from pathlib import Path

processed_dir = Path("data/Ontario/McMaster/processed")

# Load Blended output
blended_path = processed_dir / "mcmaster_lidar_rooftops_blended_3d.geojson"
print(f"Reading Blended file: {blended_path}")
if blended_path.exists():
    blended_gdf = gpd.read_file(blended_path)
    print("Blended GeoDataFrame columns:", blended_gdf.columns.tolist())
    print("Blended row count:", len(blended_gdf))
    print("Blended types count:\n", blended_gdf['type'].value_counts() if 'type' in blended_gdf.columns else "No 'type' column")
    
    # Check heights for parent buildings (type == 'building')
    parents = blended_gdf[blended_gdf['type'] == 'building'] if 'type' in blended_gdf.columns else blended_gdf
    print("\n--- Blended Parent Building Heights Stats ---")
    if not parents.empty:
        if 'height' in parents.columns:
            print(parents[['height', 'height_p90', 'height_max']].describe())
            print(parents[['BUILDING', 'height', 'height_p90']].head(20))
        else:
            print("No height column in parents!")
    else:
        print("No parent building features found!")
        
    # Check heights for parts (type == 'building_part')
    if 'type' in blended_gdf.columns:
        parts = blended_gdf[blended_gdf['type'] == 'building_part']
        print("\n--- Blended Building Part Heights Stats ---")
        if not parts.empty:
            print(parts[['height', 'height_p90', 'height_max']].describe())
            print(parts[['BUILDING', 'parent_id', 'height', 'height_p90']].head(20))
        else:
            print("No building parts found!")
