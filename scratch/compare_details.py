import geopandas as gpd
import os

def compare():
    path_a = "artifacts/building_132_method_a.geojson"
    path_d = "artifacts/building_132_method_d.geojson"
    
    if os.path.exists(path_a):
        gdf_a = gpd.read_file(path_a)
        print("--- Method A Segments ---")
        print(f"Count: {len(gdf_a)}")
        print(gdf_a[['height_p90', 'clean_area', 'clean_volume']])
        print(f"Total Area: {gdf_a['clean_area'].sum():.2f}")
        print(f"Total Vol: {gdf_a['clean_volume'].sum():.2f}")
        
    if os.path.exists(path_d):
        gdf_d = gpd.read_file(path_d)
        print("\n--- Method D Segments ---")
        print(f"Count: {len(gdf_d)}")
        print(gdf_d[['height_p90', 'clean_area', 'clean_volume']])
        print(f"Total Area: {gdf_d['clean_area'].sum():.2f}")
        print(f"Total Vol: {gdf_d['clean_volume'].sum():.2f}")
        if 'clean_surface_area' in gdf_d.columns:
            print(f"Reported SA: {gdf_d['clean_surface_area'].iloc[0]:.2f}")

if __name__ == "__main__":
    compare()
