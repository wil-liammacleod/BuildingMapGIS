import geopandas as gpd
import pandas as pd
import topojson as tp
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import shapely
import os
import time

def fix_geometry(geom):
    """Try to fix invalid geometries."""
    if not geom.is_valid:
        return geom.buffer(0)
    return geom

def run_method_a(file_path, building_id, tolerance=0.5):
    """
    Plan A: Topology-Aware Simplification using TopoJSON.
    Fixes gaps by simplifying shared edges once.
    """
    start_total = time.perf_counter()
    print(f"--- Running Method A (TopoJSON) for Building {building_id} ---")
    
    start_load = time.perf_counter()
    gdf = gpd.read_file(file_path)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:2958")
    bldg = gdf[gdf['BUILDING'] == building_id].copy()
    load_time = time.perf_counter() - start_load
    
    if bldg.empty:
        print(f"Building {building_id} not found.")
        return

    # 1. Preliminary cleanup
    start_topo = time.perf_counter()
    bldg.geometry = bldg.geometry.apply(fix_geometry)
    
    # 2. Topology-Aware Simplification
    # Using 2000 (approx 10-20cm precision) to force snapping of shared edges
    bldg.geometry = bldg.geometry.buffer(0.05, join_style=2)
    print(f"Creating topology with prequantize={2000}...")
    topo = tp.Topology(bldg, prequantize=2000)
    
    print(f"Simplifying topology with epsilon={tolerance}...")
    topo_simplified = topo.toposimplify(epsilon=tolerance)
    
    # 3. Convert back to GDF for the volume and basic cleaning
    bldg_topo = topo_simplified.to_gdf()
    topo_time = time.perf_counter() - start_topo
    
    # 4. Sort by height DESC and subtract overlaps
    start_clean = time.perf_counter()
    bldg_topo = bldg_topo.sort_values(by='height_p90', ascending=False)
    
    processed_footprints = None
    cleaned_segments = []
    total_volume = 0
    
    for idx, row in bldg_topo.iterrows():
        geom = row.geometry
        height = row['height_p90']
        if processed_footprints is None:
            actual_geom = geom
            processed_footprints = geom
        else:
            try:
                actual_geom = shapely.difference(geom, processed_footprints, grid_size=0.1)
                processed_footprints = shapely.union(processed_footprints, geom, grid_size=0.1)
            except Exception as e:
                actual_geom = geom
                
        if not actual_geom.is_empty:
            actual_geom = fix_geometry(actual_geom)
            seg_area = actual_geom.area
            total_volume += seg_area * height
            
            new_row = row.copy()
            new_row.geometry = actual_geom
            new_row['clean_area'] = round(seg_area, 2)
            new_row['clean_volume'] = round(seg_area * height, 2)
            cleaned_segments.append(new_row)

    clean_gdf = gpd.GeoDataFrame(cleaned_segments, crs=gdf.crs)
    clean_time = time.perf_counter() - start_clean

    # 5. ANALYTIC SA CALCULATION (Sealed Building)
    start_sa = time.perf_counter()
    print("Calculating sealed Surface Area from shared boundaries...")
    
    total_sealed_sa = 0
    total_roof_area = clean_gdf['clean_area'].sum()
    
    # Building Footprint (Bottom)
    footprint_area = unary_union(clean_gdf.geometry).area
    
    # Start with sum of (perimeter * height)
    for idx, row in clean_gdf.iterrows():
        geom = row.geometry
        h = row['height_p90']
        total_sealed_sa += (geom.length * h)
        
    # Subtract shared walls (this is the "sealing" part)
    # For every pair of segments, subtract 2 * (shared_edge_length * min_height)
    shared_wall_reduction = 0
    # Optimize: only check overlapping/touching segments
    sindex = clean_gdf.sindex
    for i in range(len(clean_gdf)):
        seg_i = clean_gdf.iloc[i]
        # Find potential neighbors
        possible_neighbors_idx = list(sindex.intersection(seg_i.geometry.bounds))
        for j_idx in possible_neighbors_idx:
            if j_idx <= i: continue # Avoid double counting and self
            seg_j = clean_gdf.iloc[j_idx]
            
            # Use intersection with grid_size for robustness
            shared_edge = shapely.intersection(seg_i.geometry, seg_j.geometry, grid_size=0.1)
            if not shared_edge.is_empty and shared_edge.length > 0:
                shared_len = shared_edge.length
                min_h = min(seg_i['height_p90'], seg_j['height_p90'])
                # Subtract twice because it was counted for both i and j
                shared_wall_reduction += 2 * (shared_len * min_h)

    total_sealed_sa = total_sealed_sa - shared_wall_reduction + total_roof_area + footprint_area
    sa_time = time.perf_counter() - start_sa

    # 6. Export
    start_export = time.perf_counter()
    output_path = f"artifacts/building_{building_id}_method_a.geojson"
    os.makedirs("artifacts", exist_ok=True)
    clean_gdf.to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
    export_time = time.perf_counter() - start_export
    
    total_time = time.perf_counter() - start_total

    print("\n--- Method A Performance ---")
    print(f"Data Loading:   {load_time:.3f}s")
    print(f"Topo Process:   {topo_time:.3f}s")
    print(f"Overlap Clean:  {clean_time:.3f}s")
    print(f"Sealed SA Calc: {sa_time:.3f}s")
    print(f"TOTAL TIME:     {total_time:.3f}s")
    
    print("\n--- Method A Results ---")
    print(f"Total Building Volume: {total_volume:.2f} m³")
    print(f"Total Sealed Surface Area: {total_sealed_sa:.2f} m²")
    print(f"Saved to: {output_path}")
    return clean_gdf

if __name__ == "__main__":
    rooftops_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    run_method_a(rooftops_path, building_id=132, tolerance=0.5)
