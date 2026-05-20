import geopandas as gpd
import pandas as pd
import trimesh
import topojson as tp
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import shapely
import os
import numpy as np
import time

def fix_geometry(geom):
    if not geom.is_valid:
        return geom.buffer(0)
    return geom

def run_method_b(file_path, building_id, tolerance=0.5):
    """
    Plan B: 3D Mesh Conversion via Trimesh.
    Uses TopoJSON for gap fixing, then extrudes to 3D meshes.
    """
    start_total = time.perf_counter()
    print(f"--- Running Method B (Trimesh) for Building {building_id} ---")
    
    start_load = time.perf_counter()
    gdf = gpd.read_file(file_path)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:2958")
    bldg = gdf[gdf['BUILDING'] == building_id].copy()
    load_time = time.perf_counter() - start_load

    if bldg.empty:
        print(f"Building {building_id} not found.")
        return

    # 1. Gap Fixing (using TopoJSON logic from Plan A)
    start_topo = time.perf_counter()
    print("Fixing gaps with TopoJSON...")
    bldg.geometry = bldg.geometry.apply(fix_geometry)
    # Add buffer to close gaps before topology
    bldg.geometry = bldg.geometry.buffer(0.05, join_style=2)
    topo = tp.Topology(bldg, prequantize=2000)
    bldg_clean = topo.toposimplify(epsilon=tolerance).to_gdf()
    bldg_clean = bldg_clean.sort_values(by='height_p90', ascending=False)
    topo_time = time.perf_counter() - start_topo

    # Subtract overlaps
    start_clean = time.perf_counter()
    processed_footprints = None
    final_segments = []
    
    for idx, row in bldg_clean.iterrows():
        geom = row.geometry
        try:
            if processed_footprints is None:
                actual_geom = geom
                processed_footprints = geom
            else:
                actual_geom = shapely.difference(geom, processed_footprints, grid_size=0.1)
                processed_footprints = shapely.union(processed_footprints, geom, grid_size=0.1)
        except Exception as e:
            print(f"⚠️ Topology error on segment {idx}, fallback: {e}")
            actual_geom = geom
        
        if not actual_geom.is_empty:
            actual_geom = fix_geometry(actual_geom)
            new_row = row.copy()
            new_row.geometry = actual_geom
            new_row['clean_area'] = round(actual_geom.area, 2)
            new_row['clean_volume'] = round(actual_geom.area * row['height_p90'], 2)
            final_segments.append(new_row)
    clean_time = time.perf_counter() - start_clean

    # 2. Extrude to 3D
    start_mesh = time.perf_counter()
    print("Extruding segments to 3D meshes...")
    meshes = []
    for row in final_segments:
        geom = row.geometry
        height = row['height_p90']
        
        polys = []
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        elif hasattr(geom, 'geoms'):
            polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
            flat_polys = []
            for p in polys:
                if isinstance(p, MultiPolygon): flat_polys.extend(list(p.geoms))
                else: flat_polys.append(p)
            polys = flat_polys
        
        for poly in polys:
            if poly.is_empty or poly.area < 0.01: continue
            try:
                mesh = trimesh.creation.extrude_polygon(poly, height)
                meshes.append(mesh)
            except Exception as e:
                print(f"⚠️ Failed to extrude polygon: {e}")

    if not meshes:
        print("Error: No meshes created.")
        return

    combined_mesh = trimesh.util.concatenate(meshes)
    
    try:
        print("Attempting boolean union for accurate SA...")
        union_mesh = combined_mesh.union(engine='blender')
    except Exception as e:
        print(f"⚠️ Boolean union skipped or failed: {e}. Using concatenated mesh.")
        union_mesh = combined_mesh

    total_sa = union_mesh.area
    total_volume = union_mesh.volume
    is_watertight = union_mesh.is_watertight
    mesh_time = time.perf_counter() - start_mesh

    # 4. Export
    start_export = time.perf_counter()
    os.makedirs("artifacts", exist_ok=True)
    glb_path = f"artifacts/building_{building_id}.glb"
    union_mesh.export(glb_path)
    
    clean_gdf = gpd.GeoDataFrame(final_segments, crs=gdf.crs)
    geojson_path = f"artifacts/building_{building_id}_method_b.geojson"
    clean_gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")
    export_time = time.perf_counter() - start_export
    
    total_time = time.perf_counter() - start_total

    print("\n--- Method B Performance ---")
    print(f"Data Loading:   {load_time:.3f}s")
    print(f"Topo Simpl:     {topo_time:.3f}s")
    print(f"Overlap Clean:  {clean_time:.3f}s")
    print(f"Mesh Process:   {mesh_time:.3f}s")
    print(f"Export:         {export_time:.3f}s")
    print(f"TOTAL TIME:     {total_time:.3f}s")

    print("\n--- Method B Results ---")
    print(f"Total Building Volume: {total_volume:.2f} m³")
    print(f"Total Mesh Surface Area: {total_sa:.2f} m²")
    print(f"Is Watertight: {is_watertight}")
    return union_mesh

if __name__ == "__main__":
    rooftops_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    run_method_b(rooftops_path, building_id=132)
