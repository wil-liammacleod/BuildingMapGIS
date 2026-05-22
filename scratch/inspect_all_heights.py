import geopandas as gpd
from pathlib import Path

processed_dir = Path("data/Ontario/McMaster/processed")

files = {
    "Overture (Modern)": "mcmaster_overture_buildings_3d.geojson",
    "LiDAR (Auto-Extracted)": "mcmaster_buildings_3d.geojson",
    "StatCan (Legacy)": "mcmaster_statcan_buildings_3d.geojson",
    "LiDAR (Rooftop Raw)": "mcmaster_lidar_rooftops_3d.geojson",
    "LiDAR (Method D Healed)": "mcmaster_lidar_rooftops_clean_3d.geojson",
    "LiDAR (S3DB Parents & Parts)": "mcmaster_lidar_rooftops_s3db_3d.geojson",
    "LiDAR (Blended OSM)": "mcmaster_lidar_rooftops_blended_3d.geojson"
}

for name, filename in files.items():
    path = processed_dir / filename
    print(f"\n=================== {name} ({filename}) ===================")
    if path.exists():
        gdf = gpd.read_file(path)
        print("Row count:", len(gdf))
        print("Columns:", gdf.columns.tolist())
        
        # Check height columns
        for col in ['height', 'height_p90', 'height_max']:
            if col in gdf.columns:
                print(f"  {col} stats:")
                print(gdf[col].describe().to_string().replace('\n', '\n  '))
            else:
                print(f"  {col} is MISSING!")
    else:
        print("File does not exist!")
