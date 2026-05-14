import whitebox_workflows as wbw
import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys

# Import our custom modules (now in the same directory)
import importer
import api_test

def main():
    # Define the Province and City you are processing here!
    province_name = "Ontario"
    city_name = "McMaster" 
    
    # Bounding Box (Lat/Lon) [min_lon, min_lat, max_lon, max_lat]
    bbox = [-79.925, 43.255, -79.910, 43.268]
    
    print(f"🚀 Starting {city_name}, {province_name} Building Extraction Pipeline...")
    print("-" * 50)
    
    # 0. Test the Ontario GeoHub API (Optional diagnostic)
    api_test.test_ontario_elevation_api()
    print("-" * 50)
    
    # 1. Setup Province-based Project Structure
    province_dir = Path(f"./data/{province_name}")
    footprints_dir = province_dir / "footprints"
    city_raw_dir = province_dir / city_name / "raw"
    city_processed_dir = province_dir / city_name / "processed"
    
    # Create the directories
    footprints_dir.mkdir(parents=True, exist_ok=True)
    city_raw_dir.mkdir(parents=True, exist_ok=True)
    city_processed_dir.mkdir(parents=True, exist_ok=True)
    
    # File Paths 
    dsm_file = city_raw_dir / f"{city_name.lower()}_dsm.tif"
    dtm_file = city_raw_dir / f"{city_name.lower()}_dtm.tif"
    
    # --- CHECK IF FILES EXIST BEFORE RUNNING ---
    has_data = importer.ensure_data_exists(
        province_name=province_name,
        city_name=city_name, 
        footprints_dir=footprints_dir,
        dsm_path=dsm_file,
        dtm_path=dtm_file,
        bbox=bbox,
        raw_dir=city_raw_dir
    )
    
    if not has_data:
        sys.exit(1) # Stop the script here until data is downloaded

    # 2. Setup Whitebox Workflows Environment
    wbe = wbw.WbEnvironment()
    wbe.verbose = True
    
    # 3. Read the Lidar Raster Data into Memory
    print(f"\nLoading DSM and DTM into memory for {city_name}...")
    dsm = wbe.read_raster(str(dsm_file))
    dtm = wbe.read_raster(str(dtm_file))
    
    # 4. Calculate Normalized Digital Surface Model (nDSM)
    print("Calculating nDSM...")
    ndsm = dsm - dtm
    
    # Save the processed nDSM into the processed folder
    wbe.write_raster(ndsm, str(city_processed_dir / f"{city_name.lower()}_ndsm_output.tif")) 
    
    # 5. Surface Roughness Filter (Removing Trees)
    print("Applying Surface Ruggedness Filter to identify trees...")
    roughness = wbe.ruggedness_index(ndsm)
    wbe.write_raster(roughness, str(city_processed_dir / f"{city_name.lower()}_roughness_output.tif"))
    
    # 6. Load Vector Footprints
    print(f"\\nLoading Building Footprints...")
    datasets = {}
    
    # 6a. Load StatCan Footprints
    try:
        footprint_files = list(footprints_dir.glob("*.shp")) + list(footprints_dir.glob("*.gpkg"))
        if footprint_files:
            from shapely.geometry import box
            native_crs = gpd.read_file(footprint_files[0], rows=1).crs
            bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
            bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs="EPSG:4326")
            native_bbox = tuple(bbox_gdf.to_crs(native_crs).total_bounds)
            
            gdfs = []
            for fp_file in footprint_files:
                gdf = gpd.read_file(fp_file, bbox=native_bbox)
                if not gdf.empty:
                    gdfs.append(gdf)
            if gdfs:
                datasets["statcan"] = pd.concat(gdfs, ignore_index=True)
                print(f"✅ Loaded {len(datasets['statcan'])} StatCan buildings.")
    except Exception as e:
        print(f"❌ Failed to load StatCan data: {e}")
        
    # 6b. Load Overture Footprints
    overture_path = city_raw_dir / "overture_footprints.geojson"
    if overture_path.exists():
        overture_gdf = gpd.read_file(overture_path)
        datasets["overture"] = overture_gdf
        print(f"✅ Loaded {len(overture_gdf)} Overture buildings.")

    # 6c. Native LiDAR Footprint Extraction (Whitebox)
    print(f"\\nExtracting Native Footprints directly from LiDAR...")
    try:
        import rasterio
        import numpy as np
        import shapely
        
        ndsm_path = str(city_processed_dir / f"{city_name.lower()}_ndsm_output.tif")
        rough_path = str(city_processed_dir / f"{city_name.lower()}_roughness_output.tif")
        
        with rasterio.open(ndsm_path) as src:
            ndsm_data = src.read(1)
            profile = src.profile
            
        with rasterio.open(rough_path) as src:
            rough_data = src.read(1)
            
        # Mask: > 2m tall AND Roughness < 1.5 (flat roofs)
        mask_arr = (ndsm_data >= 2.0) & (rough_data < 1.5) & (ndsm_data < 150)
        
        # Terrace heights to nearest 3 meters to merge similar roof parts
        terraced = np.round(ndsm_data / 3.0) * 3.0
        terraced[~mask_arr] = 0.0
        
        terraced_path = city_processed_dir / f"{city_name.lower()}_terraced.tif"
        with rasterio.open(terraced_path, 'w', **profile) as dst:
            dst.write(terraced.astype(rasterio.float32), 1)
            
        terraced_raster = wbe.read_raster(str(terraced_path))
        vector = wbe.raster_to_vector_polygons(terraced_raster)
        
        extracted_path = str(city_processed_dir / f"{city_name.lower()}_wb_extracted.shp")
        wbe.write_vector(vector, extracted_path)
        
        lidar_gdf = gpd.read_file(extracted_path)
        lidar_gdf = lidar_gdf[lidar_gdf['VALUE'] > 0].copy()
        
        # CRITICAL FIX: Whitebox shapefiles lack a .prj file. We must set the CRS manually!
        lidar_gdf = lidar_gdf.set_crs("EPSG:26917")
        lidar_gdf = lidar_gdf.to_crs("EPSG:4326")
        
        # Remove any potential geometry artifacts or nulls
        lidar_gdf = lidar_gdf[lidar_gdf.geometry.is_valid]
        
        lidar_gdf['height_p90'] = lidar_gdf['VALUE']
        lidar_gdf['height_max'] = lidar_gdf['VALUE']
        lidar_gdf['address'] = 'LiDAR Auto-Extracted'
        lidar_gdf['type'] = 'Multi-Tiered Polygon'
        
        import shapely
        lidar_gdf.geometry = shapely.force_2d(lidar_gdf.geometry)
        
        lidar_output = city_processed_dir / f"{city_name.lower()}_lidar_buildings_3d.geojson"
        print(f"Saving final LiDAR dataset to: {lidar_output.name}...")
        lidar_gdf.to_file(lidar_output, driver="GeoJSON")
        print(f"✅ Extracted {len(lidar_gdf)} native LiDAR sub-polygons.")
    except Exception as e:
        print(f"❌ Failed to extract LiDAR footprints: {e}")

    if not datasets:
        print("❌ No building footprint datasets were loaded! Cannot proceed.")
        return

    # 7. Zonal Statistics (Extract Height per Building)
    from rasterio.mask import mask
    import numpy as np
    from tqdm import tqdm
    import shapely
    
    ndsm_path = str(city_processed_dir / f"{city_name.lower()}_ndsm_output.tif")
    
    for dataset_name, buildings_gdf in datasets.items():
        print(f"\\nExtracting heights from nDSM for {dataset_name.upper()} dataset...")
        
        # Reproject to match the LiDAR rasters
        buildings_gdf = buildings_gdf.to_crs("EPSG:26917")
        
        heights_p90 = []
        heights_max = []
        
        with rasterio.open(ndsm_path) as src:
            for geom in tqdm(buildings_gdf.geometry, desc=f"Zonal Stats ({dataset_name})", unit="bldg"):
                try:
                    out_image, _ = mask(src, [geom], crop=True, nodata=-9999)
                    valid_pixels = out_image[out_image != -9999]
                    
                    if len(valid_pixels) > 0:
                        heights_p90.append(np.percentile(valid_pixels, 90))
                        heights_max.append(np.max(valid_pixels))
                    else:
                        heights_p90.append(0.0)
                        heights_max.append(0.0)
                except ValueError:
                    heights_p90.append(0.0)
                    heights_max.append(0.0)
                    
        buildings_gdf['height_p90'] = heights_p90
        buildings_gdf['height_max'] = heights_max
        
        # Filter out buildings with 0 or negative heights (errors, or sheds < 2m)
        valid_buildings = buildings_gdf[buildings_gdf['height_p90'] >= 2.0].copy()
        print(f"Extracted heights for {len(valid_buildings)} valid {dataset_name} buildings.")
        
        # 8. Export Final Output
        valid_buildings = valid_buildings.to_crs("EPSG:4326")
        valid_buildings.geometry = shapely.force_2d(valid_buildings.geometry)
        
        final_output = city_processed_dir / f"{city_name.lower()}_{dataset_name}_buildings_3d.geojson"
        print(f"Saving final 3D dataset to: {final_output.name}...")
        valid_buildings.to_file(final_output, driver="GeoJSON")
        
    print("\\n🎉 Pipeline Complete! Both datasets processed.")

if __name__ == "__main__":
    main()
