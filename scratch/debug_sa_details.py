import geopandas as gpd
import shapely
import os

def debug_sa(path, name):
    if not os.path.exists(path):
        print(f"{name} file not found.")
        return
    gdf = gpd.read_file(path)
    if gdf.crs != "EPSG:2958":
        gdf = gdf.to_crs("EPSG:2958")
    
    total_roof_area = gdf.geometry.area.sum()
    footprint = gdf.geometry.unary_union
    footprint_area = footprint.area
    
    total_wall_area = 0.0
    for _, row in gdf.iterrows():
        total_wall_area += row.geometry.boundary.length * row['height_p90']
        
    shared_wall_reduction = 0.0
    sindex = gdf.sindex
    for i in range(len(gdf)):
        seg_i = gdf.iloc[i]
        possible_neighbors = list(sindex.intersection(seg_i.geometry.bounds))
        for j_idx in possible_neighbors:
            if j_idx <= i:
                continue
            seg_j = gdf.iloc[j_idx]
            
            # Intersection of boundaries (1D line)
            shared_edge_b = shapely.intersection(seg_i.geometry.boundary, seg_j.geometry.boundary, grid_size=0.1)
            # Intersection of polygons
            shared_edge_p = shapely.intersection(seg_i.geometry, seg_j.geometry, grid_size=0.1)
            
            min_h = min(seg_i['height_p90'], seg_j['height_p90'])
            
            if not shared_edge_b.is_empty and shared_edge_b.length > 0:
                shared_wall_reduction += 2.0 * (shared_edge_b.length * min_h)
                
            print(f"  Shared i={i}, j={j_idx}: min_h={min_h:.1f}")
            print(f"    Boundary intersection length: {shared_edge_b.length:.3f}")
            print(f"    Polygon intersection length: {shared_edge_p.length:.3f} (geom_type: {shared_edge_p.geom_type})")
            if shared_edge_p.geom_type == 'Polygon' or shared_edge_p.geom_type == 'MultiPolygon':
                print(f"    Polygon intersection area: {shared_edge_p.area:.3f}")
                
    final_wall_area = total_wall_area - shared_wall_reduction
    total_sealed_sa = total_roof_area + footprint_area + final_wall_area
    
    print(f"\n=== {name} Summary ===")
    print(f"Total Roof Area: {total_roof_area:.2f}")
    print(f"Footprint Area: {footprint_area:.2f}")
    print(f"Raw Wall Area: {total_wall_area:.2f}")
    print(f"Shared Wall Reduction: {shared_wall_reduction:.2f}")
    print(f"Final Wall Area: {final_wall_area:.2f}")
    print(f"Total Sealed SA: {total_sealed_sa:.2f}\n")

if __name__ == "__main__":
    debug_sa("artifacts/building_132_method_a.geojson", "Method A")
    debug_sa("artifacts/building_132_method_d.geojson", "Method D")
