import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import os
import time

def fix_geometry(geom):
    if not geom.is_valid:
        return geom.buffer(0)
    return geom

def run_method_c(file_path, building_id, resolution=0.25):
    """
    Plan C: Voxel Grid Rasterization.
    Eliminates topology issues by working on a regular 3D grid.
    """
    start_total = time.perf_counter()
    print(f"--- Running Method C (Voxel) for Building {building_id} ---")
    
    start_load = time.perf_counter()
    gdf = gpd.read_file(file_path)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:2958")
    bldg = gdf[gdf['BUILDING'] == building_id].copy()
    load_time = time.perf_counter() - start_load

    if bldg.empty:
        print(f"Building {building_id} not found.")
        return

    bldg.geometry = bldg.geometry.apply(fix_geometry)
    
    # 1. Setup Voxel Grid
    start_setup = time.perf_counter()
    bounds = bldg.total_bounds
    max_height = bldg['height_p90'].max()
    pad = resolution * 2
    minx, miny, maxx, maxy = bounds[0]-pad, bounds[1]-pad, bounds[2]+pad, bounds[3]+pad
    
    nx = int(np.ceil((maxx - minx) / resolution))
    ny = int(np.ceil((maxy - miny) / resolution))
    nz = int(np.ceil(max_height / resolution))
    
    print(f"Allocating voxel grid: {nx}x{ny}x{nz} ({nx*ny*nz:,} voxels)...")
    grid = np.zeros((nz, ny, nx), dtype=bool)
    transform = from_bounds(minx, miny, maxx, maxy, nx, ny)
    
    from scipy.ndimage import binary_fill_holes
    from shapely.ops import unary_union
    
    # HEALING STEP: Create a solid footprint hull
    print("Computing building hull/footprint...")
    footprint_union = unary_union([s.geometry for _, s in bldg.iterrows()])
    if hasattr(footprint_union, 'geoms'): # MultiPolygon
        solid_footprint = MultiPolygon([Polygon(p.exterior) for p in footprint_union.geoms])
    else: # Polygon
        solid_footprint = Polygon(footprint_union.exterior)
    setup_time = time.perf_counter() - start_setup

    # 2. Fill Voxels
    start_fill = time.perf_counter()
    for _, row in bldg.iterrows():
        geom = row.geometry
        height = row['height_p90']
        mask_2d = rasterize([(geom, 1)], out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8, all_touched=True).astype(bool)
        mask_2d = binary_fill_holes(mask_2d)
        z_top = int(np.ceil(height / resolution))
        grid[:z_top, mask_2d] = True

    print("Final hull-based gap filling...")
    hull_mask = rasterize([(solid_footprint, 1)], out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8, all_touched=True).astype(bool)
    for z in range(nz):
        grid[z] = binary_fill_holes(grid[z] & hull_mask) | grid[z]
    fill_time = time.perf_counter() - start_fill

    # 3. Compute Metrics
    start_metrics = time.perf_counter()
    volume = grid.sum() * (resolution**3)
    sa_faces = 0
    diff_z = np.diff(grid.astype(int), axis=0, prepend=0, append=0)
    sa_faces += np.abs(diff_z).sum()
    diff_y = np.diff(grid.astype(int), axis=1, prepend=0, append=0)
    sa_faces += np.abs(diff_y).sum()
    diff_x = np.diff(grid.astype(int), axis=2, prepend=0, append=0)
    sa_faces += np.abs(diff_x).sum()
    surface_area = sa_faces * (resolution**2)
    metrics_time = time.perf_counter() - start_metrics
    
    # 4. Export
    start_export = time.perf_counter()
    import json
    results = {
        "method": "Method C (Voxel)",
        "resolution": resolution,
        "total_volume": round(volume, 2),
        "total_surface_area": round(surface_area, 2)
    }
    with open(f"artifacts/building_{building_id}_method_c.json", "w") as f:
        json.dump(results, f)
        
    print("Generating voxel visualization GeoJSON...")
    voxel_polys = []
    from shapely.geometry import box
    solid_counts = grid.sum(axis=0)
    y_idxs, x_idxs = np.where(solid_counts > 0)
    for y_idx, x_idx in zip(y_idxs, x_idxs):
        wx = minx + x_idx * resolution
        wy = maxy - (y_idx + 1) * resolution
        poly = box(wx, wy, wx + resolution, wy + resolution)
        voxel_polys.append({'geometry': poly, 'height_p90': solid_counts[y_idx, x_idx] * resolution, 'clean_area': resolution**2})
    
    voxel_gdf = gpd.GeoDataFrame(voxel_polys, crs="EPSG:2958")
    voxel_gdf.to_crs("EPSG:4326").to_file(f"artifacts/building_{building_id}_method_c.geojson", driver="GeoJSON")
    export_time = time.perf_counter() - start_export
    
    total_time = time.perf_counter() - start_total

    print("\n--- Method C Performance ---")
    print(f"Data Loading:   {load_time:.3f}s")
    print(f"Setup/Hull:     {setup_time:.3f}s")
    print(f"Voxel Fill:     {fill_time:.3f}s")
    print(f"Metrics Calc:   {metrics_time:.3f}s")
    print(f"Export:         {export_time:.3f}s")
    print(f"TOTAL TIME:     {total_time:.3f}s")
    
    print("\n--- Method C Results ---")
    print(f"Resolution: {resolution} m")
    print(f"Total Building Volume: {volume:.2f} m³")
    print(f"Total Voxel Surface Area: {surface_area:.2f} m²")
    return volume, surface_area

if __name__ == "__main__":
    rooftops_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    run_method_c(rooftops_path, building_id=132, resolution=0.25)
