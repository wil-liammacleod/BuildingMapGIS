"""
src/import_pointcloud.py

Queries the federal CanElevation LiDAR tile index streamingly over S3 using vsicurl,
downloads the intersecting COPC LAZ tiles for a given bounding box/area,
converts them to standard LAS 1.2 format, and merges them using Whitebox Workflows.

Usage:
    uv run src/import_pointcloud.py --limit 18
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from shapely.geometry import box
import geopandas as gpd
from tqdm import tqdm
import whitebox_workflows as wbw

# Add src to python path to import convert_copc_to_standard_laz
sys.path.append(str(Path(__file__).parent.resolve()))
import importer

# Default spatial configurations
KNOWN_AREAS = {
    ("ontario", "hamilton", "ward_1"): {
        "bbox": [-79.9462, 43.2417, -79.8743, 43.2940],
        "project": "Hamilton_Niagara_2021_2",
    },
    ("ontario", "mcmaster", ""): {
        "bbox": [-79.925, 43.255, -79.910, 43.268],
        "project": "Hamilton_Niagara_2021_2",
    }
}

CAN_ELEV_INDEX_URL = "/vsicurl/https://canelevation-lidar-point-clouds.s3-ca-central-1.amazonaws.com/pointclouds_nuagespoints/Index_LiDARtiles_tuileslidar.gpkg"

def download_file(url: str, output_path: Path):
    """Downloads a file with a progress bar."""
    print(f"   Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc="   Progress",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = f.write(data)
            bar.update(size)

def merge_las_files(input_paths: list, output_path: Path):
    import laspy
    import numpy as np

    print(f"🔄 Merging {len(input_paths)} tiles correctly using laspy (adjusting offsets and scales)...")
    
    # 1. Determine the global bounds and offsets
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for path in input_paths:
        with laspy.open(path) as f:
            min_x = min(min_x, f.header.x_min)
            max_x = max(max_x, f.header.x_max)
            min_y = min(min_y, f.header.y_min)
            max_y = max(max_y, f.header.y_max)
            min_z = min(min_z, f.header.z_min)
            max_z = max(max_z, f.header.z_max)

    # 2. Create output header with offsets rounded down
    offset_x = np.floor(min_x)
    offset_y = np.floor(min_y)
    offset_z = np.floor(min_z)

    # Use 1.2 point format 1 (standard format)
    new_header = laspy.LasHeader(point_format=1, version="1.2")
    new_header.offsets = [offset_x, offset_y, offset_z]
    new_header.scales = [0.01, 0.01, 0.01]

    # Initialize writer and write points
    with laspy.open(output_path, mode='w', header=new_header) as writer:
        total_points = 0
        for path in input_paths:
            with laspy.open(path) as reader:
                for points in reader.chunk_iterator(500_000):
                    new_points = laspy.ScaleAwarePointRecord.zeros(len(points), header=new_header)
                    new_points.x = points.x
                    new_points.y = points.y
                    new_points.z = points.z
                    new_points.intensity = points.intensity
                    new_points.classification = points.classification
                    
                    if hasattr(points, 'gps_time'):
                        new_points.gps_time = points.gps_time
                    if hasattr(points, 'return_number'):
                        new_points.return_number = np.minimum(points.return_number, 5)
                    if hasattr(points, 'number_of_returns'):
                        new_points.number_of_returns = np.minimum(points.number_of_returns, 5)
                        
                    writer.write_points(new_points)
                    total_points += len(new_points)
                    
    print(f"✅ Successfully created merged pointcloud at: {output_path} ({total_points:,} points)")

def main():
    parser = argparse.ArgumentParser(description="Import and merge LiDAR point cloud tiles based on area.")
    parser.add_argument("--province", type=str, default="Ontario", help="Province name")
    parser.add_argument("--city", type=str, default="Hamilton", help="City name")
    parser.add_argument("--ward", type=str, default="Ward 1", help="Ward name (optional)")
    parser.add_argument("--limit", type=int, default=18, help="Max tiles to download (0 or negative for no limit)")
    parser.add_argument("--project", type=str, default=None, help="Filter by specific LiDAR project")
    
    args = parser.parse_args()
    
    prov_key = args.province.lower()
    city_key = args.city.lower()
    ward_slug = args.ward.lower().replace(" ", "_") if args.ward else ""
    
    # Resolve bounding box and project filter
    area_key = (prov_key, city_key, ward_slug)
    if area_key in KNOWN_AREAS:
        bbox = KNOWN_AREAS[area_key]["bbox"]
        default_project = KNOWN_AREAS[area_key]["project"]
    else:
        # Fallback to McMaster if unknown Hamilton ward, or prompt error
        print(f"⚠️ Area {args.ward}, {args.city}, {args.province} not in pre-defined configs.")
        print("   Please add it to KNOWN_AREAS or run with a known area.")
        sys.exit(1)
        
    project_filter = args.project if args.project else default_project
    
    print("=" * 70)
    print(f"🛰️  CanElevation Point Cloud Importer")
    print(f"   Area:      {args.ward if args.ward else 'None'}, {args.city}, {args.province}")
    print(f"   BBox:      {bbox}")
    print(f"   Project:   {project_filter}")
    print(f"   Limit:     {args.limit if args.limit > 0 else 'All intersecting tiles'}")
    print("=" * 70)
    
    # Setup paths
    province_dir = Path(f"./data/{args.province}")
    if ward_slug:
        raw_dir = province_dir / args.city / ward_slug / "raw"
    else:
        raw_dir = province_dir / args.city / "raw"
        
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    final_prefix = ward_slug if ward_slug else args.city.lower()
    merged_output_path = raw_dir / f"{final_prefix}.laz"
    
    if merged_output_path.exists():
        print(f"✅ Merged point cloud dataset already exists at: {merged_output_path}")
        print("   Skip importing.")
        sys.exit(0)
        
    # Query index
    print("\n🔍 Querying CanElevation LiDAR tile index...")
    try:
        gdf = gpd.read_file(CAN_ELEV_INDEX_URL, bbox=tuple(bbox))
    except Exception as e:
        print(f"❌ Failed to read tile index from S3: {e}")
        sys.exit(1)
        
    print(f"🔍 Found {len(gdf)} total intersecting tiles across all projects.")
    
    # Filter by project
    if project_filter:
        gdf = gdf[gdf["Project"] == project_filter].copy()
        print(f"🔍 Filtered to {len(gdf)} tiles in project '{project_filter}'.")
        
    if gdf.empty:
        print("❌ No matching tiles found for the given bounding box and project filter.")
        sys.exit(1)
        
    # Sort tiles by distance to the bounding box center
    bbox_poly = box(*bbox)
    bbox_center = bbox_poly.centroid
    
    gdf["distance_to_center"] = gdf.geometry.centroid.distance(bbox_center)
    gdf = gdf.sort_values(by="distance_to_center").reset_index(drop=True)
    
    # Apply limit
    if args.limit > 0 and len(gdf) > args.limit:
        print(f"⚠️ Limiting download to the {args.limit} closest tiles to the study area center.")
        gdf = gdf.head(args.limit).copy()
        
    print(f"\n📋 Selected tiles to download ({len(gdf)} tiles):")
    for idx, row in gdf.iterrows():
        print(f"  [{idx+1}] {row['Tile_name']} ({row['Project']})")
        
    downloaded_paths = []
    
    # Download and convert
    for idx, row in gdf.iterrows():
        tile_name = row['Tile_name']
        copc_url = row['URL']
        
        tile_copc_path = raw_dir / f"{tile_name}.copc.laz"
        tile_standard_path = raw_dir / f"{tile_name}_standard.laz"
        
        print(f"\n📦 Processing tile {idx+1}/{len(gdf)}: {tile_name}")
        
        # 1. Download if standard and copc are missing
        if not tile_standard_path.exists():
            if not tile_copc_path.exists():
                try:
                    download_file(copc_url, tile_copc_path)
                except Exception as e:
                    print(f"❌ Failed to download {tile_name}: {e}")
                    continue
            
            # 2. Convert to standard format (laspy format compatible with Whitebox) and thin it
            try:
                importer.convert_copc_to_standard_laz(tile_copc_path, tile_standard_path)
                
                print(f"   Thinning tile to 1.0m resolution...")
                wbe = wbw.WbEnvironment()
                wbe.verbose = False
                lidar_obj = wbe.read_lidar(str(tile_standard_path))
                thinned_lidar, _ = wbe.lidar_thin(lidar_obj, resolution=1.0)
                temp_thinned = raw_dir / f"{tile_name}_thinned.laz"
                wbe.write_lidar(thinned_lidar, str(temp_thinned))
                
                # Replace standard tile with the thinned one
                tile_standard_path.unlink()
                os.rename(temp_thinned, tile_standard_path)
            except Exception as e:
                print(f"❌ Failed to convert and thin {tile_name}: {e}")
                if tile_copc_path.exists():
                    tile_copc_path.unlink()
                if tile_standard_path.exists():
                    tile_standard_path.unlink()
                continue
                
        # 3. Clean up the large COPC tile
        if tile_copc_path.exists():
            tile_copc_path.unlink()
            
        if tile_standard_path.exists():
            downloaded_paths.append(tile_standard_path)
            
    if not downloaded_paths:
        print("❌ No tiles were successfully downloaded and converted.")
        sys.exit(1)
        
    print(f"\n✅ Downloaded and converted {len(downloaded_paths)} tiles.")
    
    if len(downloaded_paths) == 1:
        print(f"🔄 Only 1 tile downloaded. Renaming to {merged_output_path.name}...")
        os.rename(downloaded_paths[0], merged_output_path)
    else:
        try:
            merge_las_files(downloaded_paths, merged_output_path)
            
            # Clean up the individual standard tiles
            print("🧹 Cleaning up individual tile files...")
            for path in downloaded_paths:
                if path.exists():
                    path.unlink()
        except Exception as e:
            print(f"❌ Failed to merge tiles: {e}")
            sys.exit(1)
            
    print("\n🏁 Pointcloud import complete!")

if __name__ == "__main__":
    main()
