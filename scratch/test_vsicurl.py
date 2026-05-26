import geopandas as gpd

print("🔍 Inspecting Hamilton_Niagara_2021_2 tiles in Ward 1 bbox...")
url = "/vsicurl/https://canelevation-lidar-point-clouds.s3-ca-central-1.amazonaws.com/pointclouds_nuagespoints/Index_LiDARtiles_tuileslidar.gpkg"
# Ward 1 bbox: [-79.9462, 43.2417, -79.8743, 43.2940]
gdf = gpd.read_file(url, bbox=(-79.9462, 43.2417, -79.8743, 43.2940))
hamilton_gdf = gdf[gdf["Project"] == "Hamilton_Niagara_2021_2"]
print(f"Number of tiles in Hamilton_Niagara_2021_2: {len(hamilton_gdf)}")
for idx, row in hamilton_gdf.iterrows():
    print(f"  {row['Tile_name']} -> {row['URL']}")
