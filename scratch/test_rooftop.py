import whitebox_workflows
import geopandas as gpd
import os

wbe = whitebox_workflows.WbEnvironment()
wbe.verbose = True

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(script_dir, 'whiteboxtest/output')
input_laz_path = os.path.join(script_dir, 'data/Ontario/McMaster/raw/ON_Niagara_20210525_NAD83CSRS_UTM17N_1km_E587_N4790_CLASS_standard.laz')
footprint_shp = os.path.join(output_dir, 'building_footprints.shp')

print("Reading LiDAR data...")
lidar = wbe.read_lidar(input_laz_path)

print("Reading building footprints...")
gdf = gpd.read_file(footprint_shp)
print(f"Loaded {len(gdf)} footprints.")

# Test cleaning options
gdf['geometry'] = gdf['geometry'].make_valid()
# Simplify with 1.0m tolerance to remove stair-steps
gdf['geometry'] = gdf['geometry'].simplify(tolerance=1.0, preserve_topology=True)
gdf = gdf.explode(index_parts=False)
gdf = gdf[gdf.geometry.type == 'Polygon'].copy()
gdf = gdf[gdf.geometry.area >= 50.0].copy()

test_shp = os.path.join(output_dir, 'test_footprints_cleaned.shp')
gdf.to_file(test_shp)
print(f"Saved {len(gdf)} cleaned footprints to {test_shp}")

test_vec = wbe.read_vector(test_shp)

print("Running LiDAR rooftop analysis with conservative parameters...")
try:
    rooftop_segments = wbe.lidar_rooftop_analysis(
        lidar_inputs=[lidar],
        building_footprints=test_vec,
        num_iterations=50,
        acceptable_model_size=80,  # Increase from 30
        search_radius=1.5         # Decrease from 2.0
    )
    print("Success! Rooftop analysis completed without panic.")
except Exception as e:
    print(f"Failed with exception: {e}")
