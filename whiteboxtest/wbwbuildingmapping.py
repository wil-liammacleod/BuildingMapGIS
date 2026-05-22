import whitebox_workflows
import os

# license_id = 'geomorphometry-2023' # Currently not needed as this script uses only license-free (free-tier) tools.

wbe = whitebox_workflows.WbEnvironment()
wbe.verbose = True # Let each of the function calls output to stdout.

print(wbe.version()) # Let's see what version of WbW we're working with

# Set up output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Script parameters
resolution = 0.5 # in meters; determines cell size of DEM/DTM
filter_size = 151 # In grid cells; will need to adjust filter size for largest building
slope_threshold = 15.0 # In degrees; 15 works well but may have to adjust if applied on steeper terrain
min_height1 = 1.5 # affects the definition of edges of features
min_height2 = 3.0 # will need to set this for minimum building height
min_area = 200 # in grid cells; at 200 grid cells, with a resoltion of 0.5 m, it would mean a building has to be at least 50 m^2.
smoothing_factor = 5 # size of smoothing filter; must be odd integer, higher applies more smoothing and set to zero for none
building_footprint_filename = os.path.join(output_dir, 'building_footprints.shp')

# Define paths
project_root = os.path.dirname(script_dir)
input_laz_path = os.path.join(project_root, 'data/Ontario/McMaster/raw/ON_Niagara_20210525_NAD83CSRS_UTM17N_1km_E587_N4790_CLASS_standard.laz')

wbe.working_directory = output_dir
print(f'Reading LiDAR data from: {input_laz_path}')

lidar = wbe.read_lidar(input_laz_path) # read in the lidar data set

# Interpolate a last-return DEM excluding vegetation (1, 3, 4, 5) and noise/bridge decks (7, 17, 18)
dem = wbe.lidar_tin_gridding(lidar, returns_included='last', cell_size=resolution, excluded_classes=[1, 3, 4, 5, 7, 17, 18])
wbe.write_raster(dem, os.path.join(output_dir, 'DEM.tif'))

# Remove the off-terrain objects (OTOs)
dtm = wbe.remove_off_terrain_objects(dem, filter_size=filter_size, slope_threshold=slope_threshold)
oto_heights = dem - dtm # measure OTO height as a DEM of diff
# wbe.write_raster(oto_heights, 'oto_heights.tif') # uncomment for quality control

# Filter out features based on height and area
otos = oto_heights > min_height1
otos = wbe.clump(otos, zero_background=True)
otos_max_hgt, tmp = wbe.zonal_statistics(oto_heights, otos, stat_type='maximum')
otos = otos_max_hgt > min_height2
# wbe.write_raster(otos, 'otos.tif') # uncomment for quality control
# Instead of the licensed generalize_classified_raster function, we filter by area using clump + zonal_statistics:
otos_clumped = wbe.clump(otos, zero_background=True)
clump_sizes, _ = wbe.zonal_statistics(otos, otos_clumped, stat_type='total', zero_is_background=True)
otos = clump_sizes >= min_area
# wbe.write_raster(otos, 'otos2.tif') # uncomment for quality control

# Save the final building raster as a TIFF
wbe.write_raster(otos, os.path.join(output_dir, 'buildings.tif'))

building_footprints = wbe.raster_to_vector_polygons(otos)
if smoothing_factor > 0:
    building_footprints = wbe.smooth_vectors(building_footprints, filter_size=smoothing_factor)

# Save the final map
wbe.write_vector(building_footprints, building_footprint_filename)

print('Done!')

# print(wbe.check_in_license(license_id)) # No license check-in required as we only use free-tier tools.