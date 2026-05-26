import os
import glob
import trimesh
import geopandas as gpd
import rasterio
import numpy as np
from tqdm import tqdm

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    glb_dir = os.path.join(output_dir, 'glb')
    
    footprint_path = os.path.join(output_dir, 'building_footprints.shp')
    dtm_path = os.path.join(output_dir, 'DTM.tif')
    combined_glb_path = os.path.join(output_dir, 'buildings_combined.glb')
    
    # Global origin for the combined coordinates
    X_global = 587000.0
    Y_global = 4790000.0
    Z_global = 0.0
    
    print("Loading building footprints and DTM...")
    if not os.path.exists(footprint_path):
        print(f"Error: Footprints shapefile not found at {footprint_path}")
        return
    if not os.path.exists(dtm_path):
        print(f"Error: DTM raster not found at {dtm_path}")
        return
        
    footprints_gdf = gpd.read_file(footprint_path)
    print(f"Loaded {len(footprints_gdf)} footprints.")
    
    # Find all individual GLB files
    glb_files = glob.glob(os.path.join(glb_dir, "building_*.glb"))
    print(f"Found {len(glb_files)} individual GLB files in {glb_dir}.")
    
    if not glb_files:
        print("No GLB files found. Please run the building mapping script first.")
        return
        
    translated_meshes = []
    
    with rasterio.open(dtm_path) as dtm_src:
        for glb_path in tqdm(glb_files, desc="Translating building meshes"):
            # Extract building ID from filename
            filename = os.path.basename(glb_path)
            try:
                bldg_id = int(filename.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                print(f"Skipping invalid filename: {filename}")
                continue
                
            if bldg_id >= len(footprints_gdf):
                print(f"Warning: Building ID {bldg_id} is out of bounds for footprints shapefile (length {len(footprints_gdf)}). Skipping.")
                continue
                
            footprint_row = footprints_gdf.iloc[bldg_id]
            footprint_poly = footprint_row.geometry
            if footprint_poly is None or footprint_poly.is_empty:
                continue
                
            X_c = footprint_poly.centroid.x
            Y_c = footprint_poly.centroid.y
            
            # Sample ground elevation Z_base
            try:
                Z_base = float(next(dtm_src.sample([(X_c, Y_c)]))[0])
                if np.isnan(Z_base) or Z_base < -100:
                    Z_base = 0.0
            except Exception:
                Z_base = 0.0
                
            try:
                # Load the individual GLB
                mesh = trimesh.load(glb_path)
                
                # If it's a Scene with multiple geometries, convert/merge into a single mesh
                if isinstance(mesh, trimesh.Scene):
                    # Concatenate geometries in scene
                    geometries = list(mesh.geometry.values())
                    if not geometries:
                        continue
                    mesh = trimesh.util.concatenate(geometries)
                
                # Translation: shift from local relative coordinate back to absolute, 
                # then offset by the global tileset origin (587000.0, 4790000.0, 0.0)
                dx = X_c - X_global
                dy = Y_c - Y_global
                dz = Z_base - Z_global
                
                mesh.apply_translation([dx, dy, dz])
                translated_meshes.append(mesh)
            except Exception as e:
                print(f"Error loading/translating {filename}: {e}")
                
    if not translated_meshes:
        print("No meshes were successfully processed.")
        return
        
    print(f"Concatenating {len(translated_meshes)} meshes...")
    try:
        combined_mesh = trimesh.util.concatenate(translated_meshes)
        print(f"Exporting combined GLB to: {combined_glb_path}")
        combined_mesh.export(combined_glb_path)
        print("Export completed successfully!")
    except Exception as e:
        print(f"Error during concatenation/export: {e}")

if __name__ == "__main__":
    main()
