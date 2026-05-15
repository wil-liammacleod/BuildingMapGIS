import whitebox_workflows as wbw
import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys

# Import our custom modules (now in the same directory)
import importer
import api_test




def detect_utm_epsg(bbox_latlon: list) -> str:
    """
    Dynamically determine the correct NAD83(CSRS) UTM EPSG code based on the
    center longitude of the bounding box. This avoids hardcoding EPSG:2958.

    For Ontario, the mapping is:
      - UTM 15N → EPSG:2956  (~-96° to -90°)
      - UTM 16N → EPSG:2957  (~-90° to -84°)
      - UTM 17N → EPSG:2958  (~-84° to -78°)
      - UTM 18N → EPSG:2959  (~-78° to -72°)
    """
    center_lon = (bbox_latlon[0] + bbox_latlon[2]) / 2.0
    utm_zone = int((center_lon + 180) / 6) + 1

    # NAD83(CSRS) UTM zones for Canada
    # EPSG = 2955 + (zone - 14) for zones 14-22 in Canada
    nad83csrs_base = {
        14: 2955, 15: 2956, 16: 2957, 17: 2958, 18: 2959,
        19: 2960, 20: 2961, 21: 2962, 22: 2963
    }
    epsg_code = nad83csrs_base.get(utm_zone, 2958)  # Fallback to 2958
    print(f"📐 Auto-detected UTM Zone {utm_zone}N → EPSG:{epsg_code}")
    return f"EPSG:{epsg_code}"


def main():
    # Define the Province and City you are processing here!
    province_name = "Ontario"
    city_name = "McMaster" 
    
    # Bounding Box (Lat/Lon) [min_lon, min_lat, max_lon, max_lat]
    # Default McMaster bbox (will be overridden if LiDAR file is found)
    bbox = [-79.925, 43.255, -79.910, 43.268]

    # 1. Setup Province-based Project Structure
    province_dir = Path(f"./data/{province_name}")
    footprints_dir = province_dir / "footprints"
    city_raw_dir = province_dir / city_name / "raw"
    city_processed_dir = province_dir / city_name / "processed"
    
    # Create directories early so we can check for files
    footprints_dir.mkdir(parents=True, exist_ok=True)
    city_raw_dir.mkdir(parents=True, exist_ok=True)
    city_processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting {city_name}, {province_name} Building Extraction Pipeline...")
    print("-" * 50)
    
    # 0. Test the Ontario GeoHub API (Optional diagnostic)
    api_test.test_ontario_elevation_api()
    print("-" * 50)
    
    # =========================================================================
    # 1a. Detect & Convert LiDAR tile
    # =========================================================================
    # Look for .laz files. If the only file is a .copc.laz, convert it first.
    laz_files = list(city_raw_dir.glob("*.laz"))
    
    # Check if we need COPC conversion
    copc_files = [f for f in laz_files if f.name.endswith(".copc.laz")]
    standard_files = [f for f in laz_files if not f.name.endswith(".copc.laz")]
    
    if copc_files and not standard_files:
        # We only have COPC files — convert the first one
        copc_path = copc_files[0]
        # Generate a clean output name
        standard_name = copc_path.name.replace(".copc.laz", "_standard.laz")
        standard_path = city_raw_dir / standard_name
        importer.convert_copc_to_standard_laz(copc_path, standard_path)
        # Refresh the file list
        laz_files = list(city_raw_dir.glob("*.laz"))
        standard_files = [f for f in laz_files if not f.name.endswith(".copc.laz")]
    
    # Prefer standard LAZ files over COPC
    lidar_file = standard_files[0] if standard_files else (laz_files[0] if laz_files else None)
    
    # Dynamically determine CRS from the LiDAR tile
    lidar_crs = None
    
    if lidar_file:
        print(f"\n📦 Using LiDAR tile: {lidar_file.name}")
        wbe_init = wbw.WbEnvironment()
        lidar_temp = wbe_init.read_lidar(str(lidar_file))
        
        num_points = lidar_temp.header.number_of_points
        print(f"   Points in file: {num_points:,}")
        
        if num_points == 0:
            print("⚠️  WARNING: Whitebox read 0 points from this file!")
            print("   If this is a .copc.laz file, conversion should have happened above.")
            print("   Continuing with raster-only pipeline...")
            lidar_file = None
        else:
            # Get bounds in native UTM
            l_min_x, l_max_x = lidar_temp.header.min_x, lidar_temp.header.max_x
            l_min_y, l_max_y = lidar_temp.header.min_y, lidar_temp.header.max_y
            
            # Convert to Lat/Lon to sync all data sources
            from shapely.geometry import box
            bounds_poly = box(l_min_x, l_min_y, l_max_x, l_max_y)
            
            # First, try a reasonable guess for the CRS based on the raw UTM bounds
            # We'll determine the correct CRS from the lat/lon bbox after a preliminary transform
            # For the initial guess, use EPSG:2958 (most common for Ontario)
            bounds_gdf = gpd.GeoDataFrame({'geometry': [bounds_poly]}, crs="EPSG:2958")
            latlon_bounds = bounds_gdf.to_crs("EPSG:4326").total_bounds
            
            bbox = [latlon_bounds[0], latlon_bounds[1], latlon_bounds[2], latlon_bounds[3]]
            
            # Now dynamically detect the correct CRS from the lat/lon center
            lidar_crs = detect_utm_epsg(bbox)
            
            # If the detected CRS differs from our initial guess, redo the bbox
            if lidar_crs != "EPSG:2958":
                bounds_gdf = gpd.GeoDataFrame({'geometry': [bounds_poly]}, crs=lidar_crs)
                latlon_bounds = bounds_gdf.to_crs("EPSG:4326").total_bounds
                bbox = [latlon_bounds[0], latlon_bounds[1], latlon_bounds[2], latlon_bounds[3]]
            
            print(f"🎯 Study area synced to LiDAR tile: {bbox}")
    
    # If no CRS was detected from LiDAR, derive it from the default bbox
    if lidar_crs is None:
        lidar_crs = detect_utm_epsg(bbox)
    
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

    # 4.5. Feature Preserving Smoothing
    print("Applying Feature Preserving Smoothing to nDSM...")
    smoothed_ndsm = wbe.feature_preserving_smoothing(ndsm, filter_size=11, normal_diff_threshold=15.0)
    wbe.write_raster(smoothed_ndsm, str(city_processed_dir / f"{city_name.lower()}_smoothed_ndsm.tif"))
    
    # 5. Surface Roughness Filter (Removing Trees)
    print("Applying Surface Ruggedness Filter to identify trees...")
    roughness = wbe.ruggedness_index(smoothed_ndsm)
    wbe.write_raster(roughness, str(city_processed_dir / f"{city_name.lower()}_roughness_output.tif"))
    
    # 6. Load Vector Footprints
    print(f"\nLoading Building Footprints...")
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
                # Ensure it has a CRS set correctly
                if datasets["statcan"].crs is None:
                    datasets["statcan"].set_crs(native_crs, inplace=True, allow_override=True)
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
    print(f"\nExtracting Native Footprints directly from LiDAR...")
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
            
        # Mask: > 2m tall AND Roughness < 1.0 (strict flat roofs to avoid trees)
        mask_arr = (ndsm_data >= 2.0) & (rough_data < 1.0) & (ndsm_data < 150)
        
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
        
        # Whitebox shapefiles lack a .prj file. Set the CRS manually using the detected CRS.
        lidar_gdf = lidar_gdf.set_crs(lidar_crs, allow_override=True)
        lidar_gdf = lidar_gdf.to_crs("EPSG:4326")
        
        # Remove any potential geometry artifacts or nulls
        lidar_gdf = lidar_gdf[lidar_gdf.geometry.is_valid]
        
        # Filter out small fragmented polygons (often remaining tree canopy)
        lidar_gdf = lidar_gdf[lidar_gdf.to_crs(lidar_crs).geometry.area >= 25.0]
        
        lidar_gdf['height_p90'] = lidar_gdf['VALUE']
        lidar_gdf['height_max'] = lidar_gdf['VALUE']
        lidar_gdf['address'] = 'LiDAR Auto-Extracted'
        lidar_gdf['type'] = 'Multi-Tiered Polygon'
        
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
        print(f"\nExtracting heights from nDSM for {dataset_name.upper()} dataset...")
        
        # Reproject to match the LiDAR rasters
        buildings_gdf = buildings_gdf.to_crs(lidar_crs)
        
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

    # =========================================================================
    # 9. LiDAR Rooftop Analysis (Phase 3: High-Detail Pass)
    # =========================================================================
    if lidar_file:
        print("\n🏗️ Starting Phase 3: High-Detail LiDAR Rooftop Analysis...")
        try:
            # Load the point cloud for processing
            print("Reading point cloud data...")
            lidar_data = wbe.read_lidar(str(lidar_file))
            print(f"   Loaded {lidar_data.header.number_of_points:,} points")
            
            if lidar_data.header.number_of_points == 0:
                print("⚠️  Cannot run rooftop analysis — 0 points in LiDAR file.")
            else:
                # Choose footprints: prefer Overture (cleaner), fallback to StatCan
                footprint_source = None
                if "overture" in datasets:
                    footprint_source = "overture"
                elif "statcan" in datasets:
                    footprint_source = "statcan"
                
                if footprint_source is None:
                    print("⚠️  No footprint datasets available for rooftop analysis. Skipping Phase 3.")
                else:
                    print(f"Running lidar_rooftop_analysis using {footprint_source.upper()} footprints...")
                    
                    # Reproject footprints to match LiDAR CRS
                    footprints_gdf = datasets[footprint_source].to_crs(lidar_crs)
                    
                    # Clip footprints to the LiDAR tile extent to avoid out-of-bounds errors
                    from shapely.geometry import box
                    l_bbox = box(
                        lidar_data.header.min_x, lidar_data.header.min_y,
                        lidar_data.header.max_x, lidar_data.header.max_y
                    )
                    footprints_gdf = footprints_gdf[footprints_gdf.intersects(l_bbox)].copy()
                    print(f"   {footprint_source.upper()} footprints intersecting LiDAR tile: {len(footprints_gdf)}")
                    
                    if len(footprints_gdf) == 0:
                        print("⚠️  No footprints overlap this LiDAR tile. Skipping Phase 3.")
                    else:
                        # Write footprints to a temp shapefile for Whitebox
                        footprints_path = city_processed_dir / "temp_footprints.shp"
                        footprints_gdf.to_file(footprints_path)
                        
                        footprints_vec = wbe.read_vector(str(footprints_path))
                        
                        # Run the rooftop analysis
                        rooftops = wbe.lidar_rooftop_analysis(
                            lidar_inputs=[lidar_data],
                            building_footprints=footprints_vec,
                            num_iterations=50
                        )
                        
                        rooftops_shp = city_processed_dir / f"{city_name.lower()}_lidar_rooftops.shp"
                        wbe.write_vector(rooftops, str(rooftops_shp))
                        
                        # Convert results to 4326 GeoJSON for visualization
                        rooftops_gdf = gpd.read_file(rooftops_shp)
                        rooftops_gdf = rooftops_gdf.set_crs(lidar_crs, allow_override=True)
                        
                        # Filter out very small artifacts
                        rooftops_gdf = rooftops_gdf[rooftops_gdf.geometry.area >= 15.0].copy()
                        
                        # ----------------------------------------------------------
                        # Compute actual height above ground (MAX_ELEV - DTM)
                        # MAX_ELEV is absolute elevation; we need to subtract the
                        # ground elevation from the DTM to get building height.
                        # ----------------------------------------------------------
                        import rasterio
                        import numpy as np
                        
                        dtm_path = str(city_raw_dir / f"{city_name.lower()}_dtm.tif")
                        with rasterio.open(dtm_path) as dtm_src:
                            # Sample ground elevation at each segment centroid
                            centroids = rooftops_gdf.geometry.centroid
                            coords = list(zip(centroids.x, centroids.y))
                            ground_elevs = np.array([val[0] for val in dtm_src.sample(coords)])
                        
                        rooftops_gdf['ground_elev'] = ground_elevs
                        rooftops_gdf['height_p90'] = (rooftops_gdf['MAX_ELEV'] - rooftops_gdf['ground_elev']).clip(lower=0).round(1)
                        rooftops_gdf['height_max'] = rooftops_gdf['height_p90']
                        
                        # Filter out segments with negligible height (ground-level artifacts)
                        rooftops_gdf = rooftops_gdf[rooftops_gdf['height_p90'] >= 2.0].copy()
                        
                        # Add metadata for app.py compatibility
                        rooftops_gdf['address'] = 'Building ' + rooftops_gdf['BUILDING'].astype(str)
                        rooftops_gdf['type'] = 'Planar Roof Segment'
                        
                        # Reproject to WGS84 for visualization
                        rooftops_gdf = rooftops_gdf.to_crs("EPSG:4326")
                        
                        print(f"   {len(rooftops_gdf)} roof segments across {rooftops_gdf['BUILDING'].nunique()} buildings")
                        
                        rooftops_output = city_processed_dir / f"{city_name.lower()}_lidar_rooftops_3d.geojson"
                        rooftops_gdf.to_file(rooftops_output, driver="GeoJSON")
                        print(f"✅ High-detail rooftops saved to: {rooftops_output.name}")
                        
                        # Cleanup temp files
                        for p in city_processed_dir.glob("temp_footprints.*"):
                            p.unlink(missing_ok=True)
                            
        except Exception as e:
            print(f"❌ LiDAR Rooftop Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n🎉 Pipeline Complete! Comparative datasets processed.")

if __name__ == "__main__":
    main()
