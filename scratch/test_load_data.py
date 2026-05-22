import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

processed_dir = Path("data/Ontario/McMaster/processed")
s3db_path = processed_dir / "mcmaster_lidar_rooftops_s3db_3d.geojson"

def load_data(path):
    if not path.exists():
        return None
    
    # Simulating _get_file_size_mb
    file_size_mb = path.stat().st_size / (1024 * 1024)
    needs_simplification = file_size_mb > 10 and "clean" not in path.name
    
    gdf = gpd.read_file(path)
    
    if needs_simplification:
        original_crs = gdf.crs
        gdf = gdf.to_crs("EPSG:2958")
        gdf = gdf[gdf.geometry.area >= 10.0].copy()
        gdf.geometry = gdf.geometry.simplify(tolerance=0.5, preserve_topology=True)
        gdf = gdf.to_crs(original_crs)
        keep_cols = {'geometry', 'height_p90', 'height_max', 'address', 'type',
                     'AVE_HGT', 'SLOPE', 'ASPECT', 'AREA', 'VALUE',
                     'num_floors', 'internal_area_sqft', 'internal_area_m2',
                     'clean_area', 'clean_volume', 'clean_volume_total',
                     'clean_surface_area', 'total_internal_sqft',
                     'max_floors', 'bldg_use', 'floor_height', 'BUILDING',
                     'lqs', 'q_coverage', 'q_canopy', 'tree_percentage',
                     'is_fallback', 'osm_id', 'parent_id', 'height'}
        drop_cols = [c for c in gdf.columns if c not in keep_cols]
        if drop_cols:
            gdf = gdf.drop(columns=drop_cols)
    
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
            
    if 'lqs' not in gdf.columns:
        gdf['lqs'] = np.nan
        
    gdf['height_p90'] = gdf['height_p90'].round(1)
    gdf['height_max'] = gdf['height_max'].round(1)
    
    gdf = gdf.explode(index_parts=False)
    return gdf

gdf = load_data(s3db_path)
print("Columns after load_data:", gdf.columns.tolist())
print("Unique building types:", gdf['type'].unique())
print("Null values count:")
print(gdf.isnull().sum())

# Let's check parent_id type in gdf
if 'parent_id' in gdf.columns:
    print("parent_id data type:", gdf['parent_id'].dtype)
    print("Unique parent_ids:", gdf['parent_id'].unique()[:20])

if 'BUILDING' in gdf.columns:
    print("BUILDING data type:", gdf['BUILDING'].dtype)
    print("Unique BUILDING values:", gdf['BUILDING'].unique()[:20])

# Separate parents and parts
parents_gdf = gdf[gdf['type'] == 'building']
parts_gdf = gdf[gdf['type'] == 'building_part']
print("Parents count:", len(parents_gdf))
print("Parts count:", len(parts_gdf))

# Check has_parts_parent_ids matching
has_parts_parent_ids = set(parts_gdf['parent_id'].unique() if 'parent_id' in parts_gdf.columns else parts_gdf['BUILDING'].unique())
print("has_parts_parent_ids:", sorted(list(has_parts_parent_ids))[:20])

unmatched_parents = parents_gdf[~parents_gdf['BUILDING'].isin(has_parts_parent_ids) | (parents_gdf['BUILDING'] == -1)]
print("Unmatched parents count:", len(unmatched_parents))
print("Is BUILDING type float or int or object in parents?", parents_gdf['BUILDING'].dtype)
print("Is parent_id type float or int or object in parts?", parts_gdf['parent_id'].dtype)

# Check elements of set has_parts_parent_ids
print("Type of elements in has_parts_parent_ids set:", type(next(iter(has_parts_parent_ids))))
print("Type of elements in parents_gdf['BUILDING']:", type(parents_gdf['BUILDING'].iloc[0]))
