import whitebox_workflows
import os
import geopandas as gpd
import rasterio
import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon, MultiPolygon

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

# Dynamically resolve standard LAZ path for the active ward
# Falls back to McMaster if not found
active_ward_raw_dir = os.path.join(project_root, 'data/Ontario/Hamilton/ward_1/raw')
standard_laz_files = []
if os.path.exists(active_ward_raw_dir):
    standard_laz_files = [
        os.path.join(active_ward_raw_dir, f) for f in os.listdir(active_ward_raw_dir)
        if f.endswith('.laz') and not f.endswith('.copc.laz')
    ]

if standard_laz_files:
    input_laz_path = standard_laz_files[0]
else:
    input_laz_path = os.path.join(project_root, 'data/Ontario/McMaster/raw/ON_Niagara_20210525_NAD83CSRS_UTM17N_1km_E587_N4790_CLASS_standard.laz')

wbe.working_directory = output_dir
print(f'Reading LiDAR data from: {input_laz_path}')

lidar = wbe.read_lidar(input_laz_path) # read in the lidar data set

# Interpolate a last-return DEM excluding vegetation (1, 3, 4, 5) and noise/bridge decks (7, 17, 18)
dem = wbe.lidar_tin_gridding(lidar, returns_included='last', cell_size=resolution, excluded_classes=[1, 3, 4, 5, 7, 17, 18])
wbe.write_raster(dem, os.path.join(output_dir, 'DEM.tif'))

# Remove the off-terrain objects (OTOs)
dtm = wbe.remove_off_terrain_objects(dem, filter_size=filter_size, slope_threshold=slope_threshold)
wbe.write_raster(dtm, os.path.join(output_dir, 'DTM.tif'))

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

# Fix geometry validity issues in Python before passing to rooftop analysis
print("Repairing building footprint geometries using GeoPandas...")
gdf_fp = gpd.read_file(building_footprint_filename)
gdf_fp['geometry'] = gdf_fp['geometry'].make_valid()
# Simplify to remove stair-stepping and clean boundaries
gdf_fp['geometry'] = gdf_fp['geometry'].simplify(tolerance=1.0, preserve_topology=True)
gdf_fp = gdf_fp.explode(index_parts=False)
gdf_fp = gdf_fp[gdf_fp.geometry.type == 'Polygon'].copy()
gdf_fp = gdf_fp[gdf_fp.geometry.area >= 50.0].copy()

# Save cleaned shapefile
cleaned_footprint_filename = os.path.join(output_dir, 'building_footprints_cleaned.shp')
gdf_fp.to_file(cleaned_footprint_filename)

# Read cleaned vector back into Whitebox Workflows
building_footprints_cleaned = wbe.read_vector(cleaned_footprint_filename)

# Run LiDAR rooftop analysis to segment buildings into facets
print("Running LiDAR rooftop analysis...")
rooftop_segments = wbe.lidar_rooftop_analysis(
    lidar_inputs=[lidar],
    building_footprints=building_footprints_cleaned,
    num_iterations=50,
    acceptable_model_size=80,
    search_radius=1.5
)
rooftops_filename = os.path.join(output_dir, 'rooftop_segments.shp')
wbe.write_vector(rooftop_segments, rooftops_filename)

# -------------------------------------------------------------------------
# Helper Functions for 3D Mesh Generation & Sloped Extrusion
# -------------------------------------------------------------------------

def extract_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    elif isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    elif hasattr(geom, 'geoms'):
        polys = []
        for g in geom.geoms:
            polys.extend(extract_polygons(g))
        return polys
    return []

def seal_holes(geom, max_area=50.0):
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        interiors_to_keep = []
        for interior in geom.interiors:
            hole_poly = Polygon(interior)
            if hole_poly.area >= max_area:
                interiors_to_keep.append(interior)
        return Polygon(geom.exterior, interiors_to_keep)
    elif isinstance(geom, MultiPolygon):
        parts = [seal_holes(p, max_area) for p in geom.geoms]
        return MultiPolygon(parts)
    return geom

def extrude_segment_with_slope(poly, slope_deg, aspect_deg, max_elev, X_c, Y_c, Z_base):
    polys = extract_polygons(poly)
    meshes = []
    
    for p in polys:
        if p.is_empty or p.area < 0.1:
            continue
            
        try:
            # Extrude to a unit height, then deform
            mesh = trimesh.creation.extrude_polygon(p, height=1.0)
            vertices = mesh.vertices.copy()
            
            # Precalculate slope params
            slope_rad = np.radians(slope_deg)
            aspect_rad = np.radians(aspect_deg)
            tan_slope = np.tan(slope_rad)
            sin_aspect = np.sin(aspect_rad)
            cos_aspect = np.cos(aspect_rad)
            
            # Find the plane constant C by evaluating at exterior ring vertices
            coords = np.array(p.exterior.coords)
            f_vals = -tan_slope * (coords[:, 0] * sin_aspect + coords[:, 1] * cos_aspect)
            f_max = np.max(f_vals)
            C = max_elev - f_max
            
            # Deform vertices relative to building centroid (X_c, Y_c) and ground Z_base
            for i in range(len(vertices)):
                vx, vy, vz = vertices[i]
                if vz < 0.5:
                    # Bottom vertices: Flat base at ground (relative Z = 0)
                    vertices[i, 0] = vx - X_c
                    vertices[i, 1] = vy - Y_c
                    vertices[i, 2] = 0.0
                else:
                    # Top vertices: Sloped plane height
                    z_roof = -tan_slope * (vx * sin_aspect + vy * cos_aspect) + C
                    z_rel = z_roof - Z_base
                    z_rel = max(1.5, z_rel) # Minimum height of 1.5m to prevent negative meshes
                    
                    vertices[i, 0] = vx - X_c
                    vertices[i, 1] = vy - Y_c
                    vertices[i, 2] = z_rel
                    
            mesh.vertices = vertices
            mesh.fix_normals()
            
            # Color faces: slate gray roof, off-white walls, dark gray bottom
            face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
            for j, normal in enumerate(mesh.face_normals):
                if normal[2] < -0.5:
                    face_colors[j] = [40, 40, 40, 255]
                elif normal[2] > 0.1:
                    face_colors[j] = [80, 100, 120, 255] # Premium Slate Gray
                else:
                    face_colors[j] = [240, 240, 235, 255] # Premium Off-white
            mesh.visual.face_colors = face_colors
            
            meshes.append(mesh)
        except Exception as e:
            print(f"Error extruding sub-polygon: {e}")
            
    return meshes

# -------------------------------------------------------------------------
# Main 3D Mesh Generation & GLB Export Loop
# -------------------------------------------------------------------------

print("Generating 3D GLB meshes for buildings...")
footprints_gdf = gpd.read_file(building_footprint_filename)
rooftops_gdf = gpd.read_file(rooftops_filename)

if footprints_gdf.crs is None:
    footprints_gdf = footprints_gdf.set_crs("EPSG:2958")
if rooftops_gdf.crs is None:
    rooftops_gdf = rooftops_gdf.set_crs("EPSG:2958")

# Map indexes to BUILDING ID
footprints_gdf['BUILDING'] = footprints_gdf.index

glb_output_dir = os.path.join(output_dir, 'glb')
os.makedirs(glb_output_dir, exist_ok=True)

with rasterio.open(os.path.join(output_dir, 'DEM.tif')) as dem_src, \
     rasterio.open(os.path.join(output_dir, 'DTM.tif')) as dtm_src:
     
     for idx, footprint_row in footprints_gdf.iterrows():
         building_id = idx
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
             
         bldg_segs = rooftops_gdf[rooftops_gdf['BUILDING'] == building_id].copy()
         
         building_meshes = []
         sloped_geoms = []
         
         # 1. Process sloped segments
         for seg_idx, seg_row in bldg_segs.iterrows():
             seg_poly = seg_row.geometry
             if seg_poly is None or seg_poly.is_empty:
                 continue
                 
             sloped_geoms.append(seg_poly)
             
             slope_deg = float(seg_row.get('SLOPE', 0.0))
             aspect_deg = float(seg_row.get('ASPECT', 0.0))
             max_elev = float(seg_row.get('MAX_ELEV', Z_base + 5.0))
             
             meshes = extrude_segment_with_slope(seg_poly, slope_deg, aspect_deg, max_elev, X_c, Y_c, Z_base)
             building_meshes.extend(meshes)
             
         # 2. Compute gaps: footprint minus union of sloped segments (healed for small objects/HVAC)
         if sloped_geoms:
             # Union of sloped segments
             try:
                 sloped_union = shapely.unary_union(sloped_geoms)
                 # Seal small holes (HVAC voids, noise)
                 sloped_union_healed = seal_holes(sloped_union, max_area=50.0)
                 gap_poly = footprint_poly.difference(sloped_union_healed)
             except Exception:
                 gap_poly = footprint_poly
         else:
             gap_poly = footprint_poly
             
         # 3. Extrude remaining gaps as flat segments
         if gap_poly is not None and not gap_poly.is_empty:
             polys = extract_polygons(gap_poly)
             for p in polys:
                 if p.area < 0.5: # Skip tiny slivers
                     continue
                     
                 try:
                     Z_roof_gap = float(next(dem_src.sample([(p.centroid.x, p.centroid.y)]))[0])
                     if np.isnan(Z_roof_gap) or Z_roof_gap < -100:
                         Z_roof_gap = Z_base + 3.0
                 except Exception:
                     Z_roof_gap = Z_base + 3.0
                     
                 Z_roof_gap = max(Z_base + 1.5, Z_roof_gap)
                 
                 meshes = extrude_segment_with_slope(p, slope_deg=0.0, aspect_deg=0.0, max_elev=Z_roof_gap, X_c=X_c, Y_c=Y_c, Z_base=Z_base)
                 building_meshes.extend(meshes)
                 
         # 4. Save to GLB
         if building_meshes:
             try:
                 combined_mesh = trimesh.util.concatenate(building_meshes)
                 glb_path = os.path.join(glb_output_dir, f"building_{building_id}.glb")
                 combined_mesh.export(glb_path)
                 print(f"Exported building {building_id} with {len(building_meshes)} segments to {glb_path}")
             except Exception as e:
                 print(f"Error exporting mesh for building {building_id}: {e}")

print('Done!')