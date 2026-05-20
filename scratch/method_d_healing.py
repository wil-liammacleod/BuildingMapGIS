import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
import shapely
import shapely.geometry
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
import topojson as tp
import os
import time

def fix_geometry(geom):
    """Try to fix invalid geometries."""
    if not geom.is_valid:
        return geom.buffer(0)
    return geom

def seal_polygon_holes(geom, max_hole_area=200.0):
    """
    Seals internal holes/voids in a polygon or multipolygon.
    Only fills holes that are smaller than max_hole_area to preserve large courtyards.
    """
    if geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        interiors_to_keep = []
        for interior in geom.interiors:
            hole_poly = Polygon(interior)
            if hole_poly.area >= max_hole_area:
                interiors_to_keep.append(interior)
        return Polygon(geom.exterior, interiors_to_keep)
    elif isinstance(geom, MultiPolygon):
        parts = [seal_polygon_holes(p, max_hole_area) for p in geom.geoms]
        return MultiPolygon(parts)
    return geom

def run_method_d(file_path, building_id, resolution=0.25, tolerance=0.3, max_hole_area=200.0,
                 residential_floor_height=3.0, commercial_floor_height=4.0, default_floor_height=3.5,
                 height_round_step=None):
    """
    Method D: Raster-based Distance Transform / Nearest-Neighbor Hole-Healing and Re-vectorization.
    Now supports floor-height classification, internal square footage estimation, and height rounding.
    """
    start_total = time.perf_counter()
    print(f"--- Running Method D (Healing & Re-vectorization) for Building {building_id} ---")
    print(f"Params: Resolution={resolution}m, Rounding={height_round_step}m, ResFloor={residential_floor_height}m, ComFloor={commercial_floor_height}m")
    
    # 1. Load Data
    start_load = time.perf_counter()
    gdf = gpd.read_file(file_path)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:2958")
    bldg = gdf[gdf['BUILDING'] == building_id].copy()
    load_time = time.perf_counter() - start_load
    
    if bldg.empty:
        print(f"Building {building_id} not found.")
        return None, {}
        
    print(f"Loaded {len(bldg)} roof segments.")
    
    # Determine building floor height based on subtype/class/name
    bldg_class = bldg['class'].iloc[0] if 'class' in bldg.columns and not pd.isna(bldg['class'].iloc[0]) else ''
    bldg_subtype = bldg['subtype'].iloc[0] if 'subtype' in bldg.columns and not pd.isna(bldg['subtype'].iloc[0]) else ''
    bldg_names = bldg['names'].iloc[0] if 'names' in bldg.columns and not pd.isna(bldg['names'].iloc[0]) else ''
    
    # Standardize string checking
    bldg_class_str = str(bldg_class).lower()
    bldg_subtype_str = str(bldg_subtype).lower()
    bldg_name_str = str(bldg_names).lower()
    
    res_indicators = {'residential', 'apartments', 'detached', 'house', 'duplex'}
    com_indicators = {'commercial', 'retail', 'office', 'school', 'university', 'college', 'library', 'hospital', 'medical', 'civic', 'religious', 'church', 'hall'}
    
    is_residential = any(ind in bldg_class_str or ind in bldg_subtype_str or ind in bldg_name_str for ind in res_indicators)
    is_commercial = any(ind in bldg_class_str or ind in bldg_subtype_str or ind in bldg_name_str for ind in com_indicators)
    
    # Default to commercial if it's in McMaster university campus
    if not is_residential and not is_commercial:
        is_commercial = True
        
    if is_residential:
        floor_height = residential_floor_height
        bldg_use = "Residential"
    elif is_commercial:
        floor_height = commercial_floor_height
        bldg_use = "Commercial"
    else:
        floor_height = default_floor_height
        bldg_use = "Other/Default"
        
    print(f"Building Class: '{bldg_class}', Subtype: '{bldg_subtype}' -> Classified as {bldg_use} (Floor Height: {floor_height}m)")
    
    # Apply height rounding to merge HVAC/roof components
    if height_round_step is not None and height_round_step > 0.0:
        bldg['height_p90'] = bldg['height_p90'].apply(
            lambda h: max(height_round_step, round(float(h) / height_round_step) * height_round_step)
        )
        print(f"Heights rounded to nearest {height_round_step}m. Distinct heights remaining: {bldg['height_p90'].nunique()}")
    
    # 2. Footprint Hole-Sealing
    start_heal = time.perf_counter()
    bldg.geometry = bldg.geometry.apply(fix_geometry)
    footprint_union = bldg.geometry.union_all()
    sealed_footprint = seal_polygon_holes(footprint_union, max_hole_area=max_hole_area)
    sealed_footprint = fix_geometry(sealed_footprint)
    
    # 3. Setup Grid
    bounds = sealed_footprint.bounds  # minx, miny, maxx, maxy
    pad = resolution * 2.0
    minx, miny, maxx, maxy = bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad
    
    nx = int(np.ceil((maxx - minx) / resolution))
    ny = int(np.ceil((maxy - miny) / resolution))
    transform = from_bounds(minx, miny, maxx, maxy, nx, ny)
    
    # 4. Rasterize Segments and Sealed Footprint
    bldg_sorted = bldg.sort_values(by='height_p90', ascending=True)
    grid = np.zeros((ny, nx), dtype=np.float32)
    
    for _, row in bldg_sorted.iterrows():
        geom = row.geometry
        height = float(row['height_p90'])
        rasterize([(geom, height)], out_shape=(ny, nx), transform=transform, out=grid, fill=0, all_touched=True)
        
    # Rasterize sealed footprint mask
    footprint_mask = rasterize([(sealed_footprint, 1)], out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8, all_touched=True).astype(bool)
    
    # 5. Distance Transform Nearest-Neighbor Gap Filling
    valid_mask = (grid > 0.0)
    if np.any(valid_mask):
        _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
        grid_healed = grid[indices[0], indices[1]]
        grid_healed[~footprint_mask] = 0.0
    else:
        grid_healed = grid
        print("⚠️ Warning: No valid height segments found to rasterize!")
        
    heal_time = time.perf_counter() - start_heal
    
    # 6. Re-vectorization
    start_revector = time.perf_counter()
    mask_to_vector = (grid_healed > 0.0)
    shapes_gen = shapes(grid_healed.astype(np.float32), mask=mask_to_vector, transform=transform)
    
    polys = []
    heights = []
    for geom_dict, val in shapes_gen:
        poly = shapely.geometry.shape(geom_dict)
        if not poly.is_empty and poly.area > 0.01:
            polys.append(poly)
            heights.append(float(val))
            
    gdf_healed = gpd.GeoDataFrame({
        'height_p90': heights,
        'geometry': polys
    }, crs="EPSG:2958")
    
    # Merge segments that have the exact same height if they are adjacent
    gdf_healed = gdf_healed.dissolve(by='height_p90').reset_index()
    gdf_healed.geometry = gdf_healed.geometry.apply(fix_geometry)
    
    revector_time = time.perf_counter() - start_revector
    
    # 7. Topology-Aware Simplification
    start_topo = time.perf_counter()
    if len(gdf_healed) > 1:
        # Buffer slightly to guarantee clean snapping
        gdf_healed.geometry = gdf_healed.geometry.buffer(0.02, join_style=2).apply(fix_geometry)
        topo = tp.Topology(gdf_healed, prequantize=2000)
        topo_simplified = topo.toposimplify(epsilon=tolerance)
        gdf_simplified = topo_simplified.to_gdf()
    else:
        gdf_simplified = gdf_healed.copy()
        gdf_simplified.geometry = gdf_simplified.geometry.simplify(tolerance, preserve_topology=True)
        
    gdf_simplified.geometry = gdf_simplified.geometry.apply(fix_geometry)
    topo_time = time.perf_counter() - start_topo
    
    # 8. Analytical Volume & Sealed Surface Area & Floors Calculation
    start_metrics = time.perf_counter()
    gdf_simplified = gdf_simplified.sort_values(by='height_p90', ascending=False)
    
    total_volume = 0.0
    total_roof_area = 0.0
    
    # Calculate areas and volumes
    gdf_simplified['clean_area'] = gdf_simplified.geometry.area.round(2)
    gdf_simplified['clean_volume'] = (gdf_simplified['clean_area'] * gdf_simplified['height_p90']).round(2)
    
    # Calculate floors and internal floor square footage
    gdf_simplified['num_floors'] = gdf_simplified['height_p90'].apply(lambda h: max(1, int(round(h / floor_height))))
    gdf_simplified['internal_area_m2'] = (gdf_simplified['clean_area'] * gdf_simplified['num_floors']).round(2)
    gdf_simplified['internal_area_sqft'] = (gdf_simplified['internal_area_m2'] * 10.76391).round(2)
    
    total_volume = gdf_simplified['clean_volume'].sum()
    total_roof_area = gdf_simplified['clean_area'].sum()
    total_internal_sqft = gdf_simplified['internal_area_sqft'].sum()
    max_floors = gdf_simplified['num_floors'].max()
    
    # Bottom Area
    simplified_footprint = gdf_simplified.geometry.union_all()
    footprint_area = simplified_footprint.area
    
    # Wall Area
    total_wall_area = 0.0
    for _, row in gdf_simplified.iterrows():
        total_wall_area += row.geometry.boundary.length * row['height_p90']
        
    shared_wall_reduction = 0.0
    sindex = gdf_simplified.sindex
    for i in range(len(gdf_simplified)):
        seg_i = gdf_simplified.iloc[i]
        possible_neighbors = list(sindex.intersection(seg_i.geometry.bounds))
        for j_idx in possible_neighbors:
            if j_idx <= i:
                continue
            seg_j = gdf_simplified.iloc[j_idx]
            
            shared_edge = shapely.intersection(seg_i.geometry.boundary, seg_j.geometry.boundary, grid_size=0.1)
            if not shared_edge.is_empty and shared_edge.length > 0:
                min_h = min(seg_i['height_p90'], seg_j['height_p90'])
                shared_wall_reduction += 2.0 * (shared_edge.length * min_h)
                
    final_wall_area = total_wall_area - shared_wall_reduction
    total_sealed_sa = total_roof_area + footprint_area + final_wall_area
    metrics_time = time.perf_counter() - start_metrics
    
    # 9. Export
    start_export = time.perf_counter()
    round_label = f"round_{height_round_step}m" if height_round_step is not None else "round_none"
    output_path = f"artifacts/building_{building_id}_method_d_res_{resolution}_{round_label}.geojson"
    os.makedirs("artifacts", exist_ok=True)
    
    # Export clean simplified geometries with all calculated floor metrics
    gdf_simplified['clean_surface_area'] = round(total_sealed_sa, 2)
    gdf_simplified['clean_volume_total'] = round(total_volume, 2)
    gdf_simplified['BUILDING'] = building_id
    gdf_simplified['resolution'] = resolution
    gdf_simplified['height_round_step'] = str(height_round_step) if height_round_step else "None"
    gdf_simplified['bldg_use'] = bldg_use
    gdf_simplified['floor_height'] = floor_height
    gdf_simplified['total_internal_sqft'] = round(total_internal_sqft, 2)
    gdf_simplified['max_floors'] = int(max_floors)
    
    gdf_simplified.to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
    export_time = time.perf_counter() - start_export
    
    total_time = time.perf_counter() - start_total
    
    def count_vertices(gdf):
        total = 0
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'Polygon':
                total += len(geom.exterior.coords)
                for ring in geom.interiors:
                    total += len(ring.coords)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    total += len(poly.exterior.coords)
                    for ring in poly.interiors:
                        total += len(ring.coords)
        return total

    num_vertices = count_vertices(gdf_simplified)

    print("\n--- Method D Results ---")
    print(f"Rounding Step:             {height_round_step}m")
    print(f"Building Classification:   {bldg_use} (Floor height: {floor_height}m)")
    print(f"Estimated Max Floors:      {max_floors}")
    print(f"Estimated Internal Sq Ft:  {total_internal_sqft:,.2f} sq ft")
    print(f"Total Building Volume:     {total_volume:.2f} m³")
    print(f"Total Sealed Surface Area: {total_sealed_sa:.2f} m²")
    print(f"Number of Shapes/Segments: {len(gdf_simplified)}")
    print(f"Total Vertices:            {num_vertices}")
    print(f"Saved to: {output_path}")
    
    metrics = {
        "resolution": resolution,
        "height_round_step": height_round_step,
        "round_label": round_label,
        "floor_height": floor_height,
        "bldg_use": bldg_use,
        "max_floors": int(max_floors),
        "total_internal_sqft": round(total_internal_sqft, 2),
        "total_time_ms": round(total_time * 1000, 1),
        "footprint_heal_ms": round(heal_time * 1000, 1),
        "revectorization_ms": round(revector_time * 1000, 1),
        "topo_simplify_ms": round(topo_time * 1000, 1),
        "metrics_calc_ms": round(metrics_time * 1000, 1),
        "volume_m3": round(total_volume, 2),
        "surface_area_m2": round(total_sealed_sa, 2),
        "num_segments": len(gdf_simplified),
        "num_vertices": num_vertices
    }
    
    return gdf_simplified, metrics

if __name__ == "__main__":
    import json
    rooftops_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    
    # 4 Scenarios: resolution 0.25m, with different rounding steps: None, 1.0m, 2.0m, 3.0m
    rounding_steps = [None, 1.0, 2.0, 3.0]
    comparison_results = []
    
    for step in rounding_steps:
        _, metrics = run_method_d(
            rooftops_path, 
            building_id=132, 
            resolution=0.25, 
            tolerance=0.3, 
            max_hole_area=200.0,
            residential_floor_height=3.0,
            commercial_floor_height=4.0,
            default_floor_height=3.5,
            height_round_step=step
        )
        comparison_results.append(metrics)
        
    comp_output_path = "artifacts/building_132_round_comparison.json"
    with open(comp_output_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print(f"\nSaved rounding comparison table to: {comp_output_path}")
