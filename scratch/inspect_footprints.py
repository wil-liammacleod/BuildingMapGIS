import geopandas as gpd
from pathlib import Path

processed_dir = Path("data/Ontario/McMaster/processed")

overture_path = processed_dir / "mcmaster_overture_buildings_3d.geojson"
print(f"Reading Overture: {overture_path}")
if overture_path.exists():
    gdf = gpd.read_file(overture_path)
    print("Overture columns:", gdf.columns.tolist())
    print("Overture size:", len(gdf))
    if 'BUILDING' in gdf.columns:
        print("BUILDING unique values count:", gdf['BUILDING'].nunique())
        print("BUILDING head:", gdf['BUILDING'].head().tolist())
    else:
        print("BUILDING not in Overture columns!")
        # Let's see if there is another ID column
        for c in gdf.columns:
            if 'id' in c.lower():
                print(f"ID-like column '{c}' head:", gdf[c].head().tolist())

raw_rooftops_path = processed_dir / "mcmaster_lidar_rooftops_3d.geojson"
print(f"\nReading raw rooftops: {raw_rooftops_path}")
if raw_rooftops_path.exists():
    gdf_raw = gpd.read_file(raw_rooftops_path)
    print("Raw rooftops columns:", gdf_raw.columns.tolist())
    print("Raw rooftops BUILDING head:", gdf_raw['BUILDING'].head(10).tolist())
    print("Raw rooftops BUILDING unique values count:", gdf_raw['BUILDING'].nunique())
