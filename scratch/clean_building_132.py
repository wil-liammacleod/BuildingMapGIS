import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import shapely
import os

def fix_geometry(geom):
    """Try to fix invalid geometries."""
    if not geom.is_valid:
        return geom.buffer(0)
    return geom

def create_extruded_blocks(file_path, building_id, tolerance=0.5):
    """
    Simplifies and 'cleans' building segments to create a non-overlapping 3D block model.
    """
    gdf = gpd.read_file(file_path)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:2958")
        
    bldg = gdf[gdf['BUILDING'] == building_id].copy()
    
    if bldg.empty:
        print(f"Building {building_id} not found.")
        return

    print(f"Processing Building {building_id}...")
    
    # 1. Preliminary cleanup and simplification
    bldg.geometry = bldg.geometry.apply(fix_geometry)
    print(f"Simplifying with tolerance {tolerance}m...")
    bldg.geometry = bldg.geometry.simplify(tolerance, preserve_topology=True)
    bldg.geometry = bldg.geometry.apply(fix_geometry) # Re-fix after simplification
    
    # 2. Sort by height DESC
    bldg = bldg.sort_values(by='height_p90', ascending=False)
    
    processed_footprints = None
    cleaned_segments = []
    total_volume = 0
    total_surface_area = 0
    
    for idx, row in bldg.iterrows():
        geom = row.geometry
        height = row['height_p90']
        
        if processed_footprints is None:
            actual_geom = geom
            processed_footprints = geom
        else:
            try:
                # Use a small buffer to avoid floating point 'slivers' causing topology errors
                actual_geom = geom.difference(processed_footprints.buffer(0.01))
                processed_footprints = unary_union([processed_footprints, geom]).buffer(0)
            except Exception as e:
                print(f"⚠️  Topology error on segment {idx}, attempting fallback...")
                # Fallback: ignore the overlap if difference fails
                actual_geom = geom
                
        if not actual_geom.is_empty:
            # Final check for validity
            actual_geom = fix_geometry(actual_geom)
            
            seg_area = actual_geom.area
            seg_volume = seg_area * height
            total_volume += seg_volume
            total_surface_area += seg_area
            
            new_row = row.copy()
            new_row.geometry = actual_geom
            new_row['clean_area'] = round(seg_area, 2)
            new_row['clean_volume'] = round(seg_volume, 2)
            cleaned_segments.append(new_row)

    if not cleaned_segments:
        print("Error: No segments survived cleaning.")
        return

    clean_gdf = gpd.GeoDataFrame(cleaned_segments, crs=gdf.crs)
    
    output_path = f"artifacts/building_{building_id}_block_model.geojson"
    os.makedirs("artifacts", exist_ok=True)
    clean_gdf.to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
    
    print("\n--- Model Results ---")
    print(f"Cleaned Segments: {len(clean_gdf)}")
    print(f"Total Building Volume: {total_volume:.2f} m³")
    print(f"Total Roof Surface Area: {total_surface_area:.2f} m²")
    
    # Calculate vertex reduction
    def count_vertices(g):
        if hasattr(g, 'exterior'): return len(g.exterior.coords)
        if hasattr(g, 'geoms'): return sum(len(p.exterior.coords) for p in g.geoms if hasattr(p, 'exterior'))
        return 0

    avg_vertices = clean_gdf.geometry.apply(count_vertices).mean()
    print(f"Average Vertices per Segment: {avg_vertices:.1f} (Massive reduction!)")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    rooftops_path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
    # Using a slightly larger tolerance to ensure clean geometry
    create_extruded_blocks(rooftops_path, building_id=132, tolerance=1.0)
