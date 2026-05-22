import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

processed_dir = Path("data/Ontario/McMaster/processed")
s3db_path = processed_dir / "mcmaster_lidar_rooftops_s3db_3d.geojson"

gdf = gpd.read_file(s3db_path)

# Ensure height columns exist
if 'height_p90' not in gdf.columns:
    if 'height' in gdf.columns:
        gdf['height_p90'] = gdf['height']
    else:
        gdf['height_p90'] = 10.0
        
if 'height_max' not in gdf.columns:
    if 'height' in gdf.columns:
        gdf['height_max'] = gdf['height']
    else:
        gdf['height_max'] = 10.0

gdf['height_p90'] = gdf['height_p90'].round(1)
gdf['height_max'] = gdf['height_max'].round(1)

parents_gdf = gdf[gdf['type'] == 'building']
parts_gdf = gdf[gdf['type'] == 'building_part']

has_parts_parent_ids = set(parts_gdf['parent_id'].unique() if 'parent_id' in parts_gdf.columns else parts_gdf['BUILDING'].unique())
unmatched_parents = parents_gdf[~parents_gdf['BUILDING'].isin(has_parts_parent_ids) | (parents_gdf['BUILDING'] == -1)]
layer_3d_gdf = gpd.GeoDataFrame(pd.concat([parts_gdf, unmatched_parents], ignore_index=True), crs=gdf.crs)

print("layer_3d_gdf row count:", len(layer_3d_gdf))
print("layer_3d_gdf columns:", layer_3d_gdf.columns.tolist())
print("layer_3d_gdf height_p90 describe:\n", layer_3d_gdf['height_p90'].describe())
print("layer_3d_gdf height_p90 head(20):\n", layer_3d_gdf['height_p90'].head(20))
print("Is any height_p90 null?", layer_3d_gdf['height_p90'].isnull().sum())
