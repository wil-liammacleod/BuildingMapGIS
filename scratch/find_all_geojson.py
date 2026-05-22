import geopandas as gpd
from pathlib import Path
import datetime

workspace_dir = Path(".")
geojson_files = list(workspace_dir.glob("**/*.geojson"))

print(f"Found {len(geojson_files)} GeoJSON files:")
for path in geojson_files:
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    size_kb = path.stat().st_size / 1024
    print(f"\nFile: {path} ({size_kb:.1f} KB, Modified: {mtime})")
    try:
        gdf = gpd.read_file(path)
        print("  Row count:", len(gdf))
        print("  Columns:", gdf.columns.tolist()[:10], "... total:", len(gdf.columns))
        for col in ['height', 'height_p90', 'height_max']:
            if col in gdf.columns:
                unique_vals = gdf[col].unique()
                print(f"  {col} unique count: {len(unique_vals)}, mean: {gdf[col].mean() if len(unique_vals)>0 else 'N/A'}")
                if len(unique_vals) <= 10:
                    print(f"    Unique values: {unique_vals}")
                else:
                    print(f"    Min: {gdf[col].min()}, Max: {gdf[col].max()}")
    except Exception as e:
        print(f"  Error reading file: {e}")
