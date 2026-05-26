import requests
import zipfile
from pathlib import Path

def download_elevation_tile(bbox: list, layer_type: str, output_path: str, province_name: str):
    """
    Downloads a perfectly cropped TIFF from Ontario GeoHub using exportImage.
    """
    if province_name.lower() != "ontario":
        print(f"⚠️ Auto-download for {province_name} LiDAR is not yet configured (Only Ontario supported right now).")
        return False

    print(f"\n📡 Requesting custom {layer_type} tile from Ontario GeoHub...")
    
    service_name = "Elevation/Ontario_DSM_LidarDerived" if layer_type == "DSM" else "Elevation/Ontario_DTM_LidarDerived"
    url = f"https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/{service_name}/ImageServer/exportImage"
    
    # Calculate approximate width and height in pixels for ~1m resolution
    # (1 degree lat = ~111km, 1 degree lon at 43N = ~81km)
    width_m = int(abs(bbox[2] - bbox[0]) * 81000)
    height_m = int(abs(bbox[3] - bbox[1]) * 111111)
    
    params = {
        'f': 'image',
        'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        'bboxSR': '4326', # Bounding box is provided in Lat/Lon
        'imageSR': '26917', # Output image in NAD83 UTM Zone 17N (meters) - Ontario standard
        'size': f"{width_m},{height_m}",
        'format': 'tiff',
        'pixelType': 'F32',
        'interpolation': 'RSP_BilinearInterpolation'
    }
    
    try:
        print(f"   Downloading ~{width_m}x{height_m} pixels. This may take a minute...")
        response = requests.get(url, params=params, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"✅ Successfully saved {layer_type} to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {layer_type}: {e}")
        return False

def download_statcan_footprints(province_name: str, output_dir: Path):
    print(f"\n📡 Requesting StatCan Open Database of Buildings for {province_name}...")
    
    if province_name.lower() == "ontario":
        # Ontario is split into 3 massive zips
        urls = [
            "https://www150.statcan.gc.ca/n1/pub/34-26-0001/2018001/zip/ODB_v3_ON_1.zip",
            "https://www150.statcan.gc.ca/n1/pub/34-26-0001/2018001/zip/ODB_v3_ON_2.zip",
            "https://www150.statcan.gc.ca/n1/pub/34-26-0001/2018001/zip/ODB_v3_ON_3.zip"
        ]
    else:
        print(f"⚠️ Auto-download mapping for {province_name} StatCan data is not configured yet.")
        return False
        
    for i, url in enumerate(urls):
        zip_path = output_dir / f"{province_name.lower()}_part_{i+1}.zip"
        print(f"   Downloading Part {i+1}/{len(urls)} (This is a 1GB+ file, please be patient)...")
        
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                    f.write(chunk)
                    
            print(f"   Unzipping Part {i+1}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                
            zip_path.unlink() # Delete zip to save space
            
        except Exception as e:
            print(f"❌ Failed to download/unzip {province_name} Part {i+1}: {e}")
            return False
            
    print(f"✅ Successfully downloaded and extracted {province_name} footprints!")
    return True

def download_overture_footprints(bbox: list, output_path: Path):
    print(f"\n📡 Requesting Overture Maps Foundation building footprints...")
    print(f"   Streaming buildings directly from Amazon S3 for bounding box: {bbox}")
    try:
        import overturemaps
        # Overture expects (xmin, ymin, xmax, ymax)
        gdf = overturemaps.geodataframe("building", bbox=bbox)
        
        # Overture returns many columns; we'll drop columns that are all NA or overly complex
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        print(f"   Saving {len(gdf)} Overture buildings to {output_path.name}...")
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"✅ Successfully downloaded Overture footprints!")
        return True
    except Exception as e:
        print(f"❌ Failed to download Overture data: {e}")
        return False

def ensure_data_exists(province_name: str, city_name: str, footprints_dir: Path, dsm_path: Path, dtm_path: Path, bbox: list = None, raw_dir: Path = None, ward_name: str = None):
    """
    Checks if province-wide footprints and city-specific TIFFs exist.
    If missing, it auto-downloads everything using APIs!
    """
    missing_data = False

    # 1. Auto-Download Elevation Rasters (City Level)
    if not dsm_path.exists() and bbox:
        success = download_elevation_tile(bbox, "DSM", str(dsm_path), province_name)
        if not success: missing_data = True
            
    if not dtm_path.exists() and bbox:
        success = download_elevation_tile(bbox, "DTM", str(dtm_path), province_name)
        if not success: missing_data = True

    # 2. Check for StatCan Footprints (Province Level)
    statcan_zips = list(footprints_dir.glob("*.zip"))
    statcan_files = list(footprints_dir.glob("*.shp")) + list(footprints_dir.glob("*.gpkg"))
    
    if len(statcan_files) == 0:
        success = download_statcan_footprints(province_name, footprints_dir)
        if not success: missing_data = True

    # 3. Auto-Download Overture Footprints (City Level)
    if raw_dir and bbox:
        overture_path = raw_dir / "overture_footprints.geojson"
        if not overture_path.exists():
            success = download_overture_footprints(bbox, overture_path)
            if not success: missing_data = True
        
    if missing_data:
        return False
        
    location_str = f"{ward_name}, {city_name}" if ward_name else city_name
    print(f"\n✅ All required data files found for {location_str}, {province_name}!")
    return True

def convert_copc_to_standard_laz(input_path: Path, output_path: Path) -> Path:
    """
    Convert a COPC (Cloud Optimized Point Cloud) LAZ 1.4 file to a standard
    LAZ 1.2 file that Whitebox Workflows can read properly.

    Whitebox's Rust backend cannot parse the LAS 1.4 "extended" point count
    header used by COPC files, reporting 0 points. This function uses laspy
    to re-write the data as LAS 1.2 / point format 1, which WBW handles fine.
    """
    import laspy
    import numpy as np

    if output_path.exists():
        print(f"✅ Standard LAZ already exists: {output_path.name}")
        return output_path

    print(f"🔄 Converting COPC → standard LAZ (this may take a few minutes for large files)...")
    print(f"   Input:  {input_path.name}")
    print(f"   Output: {output_path.name}")

    with laspy.open(input_path) as reader:
        # Create a new LAS 1.2 header with point format 1 (has GPS time)
        new_header = laspy.LasHeader(point_format=1, version="1.2")
        new_header.offsets = reader.header.offsets
        new_header.scales = reader.header.scales

        with laspy.open(output_path, mode='w', laz_backend=laspy.LazBackend.LazrsParallel, header=new_header) as writer:
            total_points = 0
            # Read in 500k-point chunks to manage memory
            for points in reader.chunk_iterator(500_000):
                # Map fields from LAS 1.4 format 6 to LAS 1.2 format 1
                new_points = laspy.ScaleAwarePointRecord.zeros(len(points), header=new_header)
                new_points.x = points.x
                new_points.y = points.y
                new_points.z = points.z
                new_points.intensity = points.intensity
                new_points.classification = points.classification

                # Copy GPS time if available
                if hasattr(points, 'gps_time'):
                    new_points.gps_time = points.gps_time

                # Copy return number and number of returns (clamped to 1.2 max of 5)
                if hasattr(points, 'return_number'):
                    new_points.return_number = np.minimum(points.return_number, 5)
                if hasattr(points, 'number_of_returns'):
                    new_points.number_of_returns = np.minimum(points.number_of_returns, 5)

                writer.write_points(new_points)
                total_points += len(new_points)
                
                if total_points % 5_000_000 < 500_000:
                    print(f"   ... {total_points:,} points written")

    print(f"✅ Conversion complete: {total_points:,} points written to {output_path.name}")
    return output_path
