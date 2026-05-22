import whitebox_workflows as wbw
import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys
import shapely
from shapely.ops import unary_union
import time

# Import our custom modules (now in the same directory)
import importer
import api_test
import quality_metrics
import pipeline_cache


def fetch_osm_buildings(bbox: list, cache_file: Path = None, raw_osm_path: Path = None, force: bool = False) -> gpd.GeoDataFrame:
    """
    Queries Overpass API to fetch existing OSM buildings in the bounding box.
    Returns a GeoDataFrame with building footprints and tags.
    """
    import requests
    import json
    from shapely.geometry import Polygon
    import sys
    
    # Check if we can load from cache
    if not force and not ("--force" in sys.argv) and cache_file and raw_osm_path and raw_osm_path.exists() and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            osm_cache = cache_data.get("osm_cache", {})
            cached_bbox = osm_cache.get("bbox")
            
            def bbox_matches(b1, b2, tolerance=0.0001):
                if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
                    return False
                return all(abs(a - b) < tolerance for a, b in zip(b1, b2))
                
            if bbox_matches(bbox, cached_bbox):
                print(f"ℹ️ Loading OSM buildings from local cache: {raw_osm_path.name}")
                osm_gdf = gpd.read_file(raw_osm_path)
                return osm_gdf
        except Exception as e:
            print(f"⚠️ Failed to load OSM data from cache: {e}. Querying Overpass API...")

    print(f"\n📡 Querying Overpass API for existing OSM buildings in bbox {bbox}...")
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Convert numpy values to standard floats to ensure clean formatting
    b = [float(x) for x in bbox]
    
    # Overpass bounding box format: (min_lat, min_lon, max_lat, max_lon)
    # bbox format: [min_lon, min_lat, max_lon, max_lat]
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["building"]({b[1]},{b[0]},{b[3]},{b[2]});
      relation["building"]({b[1]},{b[0]},{b[3]},{b[2]});
    );
    out body;
    >;
    out skel qt;
    """
    
    headers = {
        'User-Agent': 'Ontario3DBuildingGIS/1.0 (https://github.com/wil-liammacleod/BuildingMapGIS; contact: liammacleod@gmail.com)'
    }
    
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Parse ways into geometry and tags
        nodes = {node['id']: (node['lon'], node['lat']) for node in data['elements'] if node['type'] == 'node'}
        
        features = []
        for element in data['elements']:
            if element['type'] == 'way' and 'tags' in element and 'building' in element['tags']:
                el_nodes = element['nodes']
                coords = [nodes[n_id] for n_id in el_nodes if n_id in nodes]
                if len(coords) >= 3:
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid:
                            tags = element['tags']
                            tags['id'] = element['id']
                            features.append({'geometry': poly, **tags})
                    except Exception:
                        pass
                        
        if features:
            osm_gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            print(f"✅ Successfully loaded {len(osm_gdf)} buildings from OpenStreetMap.")
            
            # Save raw OSM cache and update the bbox in cache_file
            if cache_file and raw_osm_path:
                try:
                    osm_gdf.to_file(raw_osm_path, driver="GeoJSON")
                    
                    cache_data = {"stages": {}}
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'r') as f:
                                cache_data = json.load(f)
                        except Exception:
                            pass
                    
                    cache_data["osm_cache"] = {
                        "bbox": bbox
                    }
                    
                    temp_file = cache_file.with_suffix('.tmp')
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    import os
                    os.replace(temp_file, cache_file)
                    print(f"✅ Cached raw OSM data to: {raw_osm_path.name}")
                except Exception as e:
                    print(f"⚠️ Failed to cache OSM data: {e}")
                    
            return osm_gdf
    except Exception as e:
        print(f"⚠️ Overpass API request failed or timed out: {e}")
        
    print("ℹ️ Overpass API is unavailable or timed out. Conflation step will proceed with dry run.")
    return None


def blend_with_osm(s3db_parents, s3db_parts, osm_gdf, utm_crs):
    """
    Conflates and blends our S3DB building data with the existing OSM buildings.
    """
    if osm_gdf is None or osm_gdf.empty:
        print("⚠️ No OSM data available for blending. Exporting S3DB dataset directly as blended output.")
        # Just return the s3db dataset
        return pd.concat([s3db_parents, s3db_parts], ignore_index=True)
        
    print(f"\n🔄 Conflating and blending our LiDAR/Method D model with OSM ({len(osm_gdf)} buildings)...")
    
    # Project both to UTM for spatial operations
    s3db_parents_utm = s3db_parents.to_crs(utm_crs)
    s3db_parts_utm = s3db_parts.to_crs(utm_crs)
    osm_gdf_utm = osm_gdf.to_crs(utm_crs)
    
    blended_features = []
    
    def parse_int(val, default):
        if val is None or pd.isna(val):
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default
            
    def parse_float(val, default):
        if val is None or pd.isna(val):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
            
    def parse_str(val, default):
        if val is None or pd.isna(val):
            return default
        return str(val)
    
    # Match using spatial indexing
    osm_sindex = osm_gdf_utm.sindex
    
    # We will keep track of which OSM buildings are matched/updated
    matched_osm_ids = set()
    
    for idx, parent in s3db_parents_utm.iterrows():
        b_id = parent['BUILDING']
        geom = parent.geometry
        
        # Find intersecting OSM buildings
        possible_matches = list(osm_sindex.intersection(geom.bounds))
        best_match_idx = None
        max_iou = 0.0
        
        for osm_idx in possible_matches:
            osm_row = osm_gdf_utm.iloc[osm_idx]
            osm_geom = osm_row.geometry
            
            # Compute Intersection-over-Union (IoU)
            try:
                intersection_area = geom.intersection(osm_geom).area
                union_area = geom.union(osm_geom).area
                iou = intersection_area / union_area if union_area > 0 else 0
                
                if iou > 0.2 and iou > max_iou:
                    max_iou = iou
                    best_match_idx = osm_idx
            except Exception:
                pass
                
        # Get the parts for this building
        bldg_parts = s3db_parts_utm[s3db_parts_utm['parent_id'] == b_id]
        
        if best_match_idx is not None and max_iou > 0.4:
            # We found a matching building in OSM!
            osm_building = osm_gdf_utm.iloc[best_match_idx]
            osm_id = osm_building['id']
            matched_osm_ids.add(osm_id)
            
            # Check if OSM building already has 3D tags
            has_osm_3d = False
            for col in ['height', 'building:levels', 'building:part']:
                if col in osm_building and not pd.isna(osm_building[col]):
                    has_osm_3d = True
                    break
                    
            if has_osm_3d:
                # Scenario A: OSM has hand-mapped 3D. We preserve OSM 3D but can enrich tags.
                print(f"   Building {b_id} (IoU={max_iou:.2f}) matches hand-mapped 3D OSM Building {osm_id}. Preserving OSM geometry.")
                # Keep OSM building footprint and parent tags, but add/update LiDAR quality tags
                parent_blended = parent.copy()
                parent_blended.geometry = osm_building.geometry
                parent_blended['osm_id'] = osm_id
                parent_blended['name'] = parse_str(osm_building.get('name'), parse_str(parent.get('name'), ''))
                h_val = parse_float(osm_building.get('height'), parent.get('height'))
                parent_blended['height'] = h_val
                parent_blended['height_p90'] = h_val
                parent_blended['height_max'] = h_val
                parent_blended['building:levels'] = parse_int(osm_building.get('building:levels'), parent.get('building:levels'))
                parent_blended['clean_area'] = round(osm_building.geometry.area, 2)
                blended_features.append(parent_blended)
                
                # Output OSM parts if they exist or just output parent (flat) if no parts
                # Since we don't fetch OSM parts here, we just preserve the parent.
            else:
                # Scenario B: OSM has a 2D footprint only. We replace/enrich.
                parent_blended = parent.copy()
                parent_blended['osm_id'] = osm_id
                # Merge tags
                for col in ['name', 'addr:street', 'addr:housenumber']:
                    if col in osm_building and not pd.isna(osm_building[col]):
                        parent_blended[col] = osm_building[col]
                blended_features.append(parent_blended)
                
                # Output our parts
                for _, part in bldg_parts.iterrows():
                    part_blended = part.copy()
                    part_blended['osm_id'] = osm_id
                    blended_features.append(part_blended)
        else:
            # Scenario C: Building is missing in OSM. Add as new building.
            blended_features.append(parent)
            for _, part in bldg_parts.iterrows():
                blended_features.append(part)
                
    # Add any OSM buildings that were NOT matched
    for idx, osm_row in osm_gdf_utm.iterrows():
        osm_id = osm_row['id']
        if osm_id not in matched_osm_ids:
            # Create a simple parent building feature
            h_val = parse_float(osm_row.get('height'), 10.0)
            levels_val = parse_int(osm_row.get('building:levels'), 3)
            osm_bldg_use = parse_str(osm_row.get('building'), 'yes').title()
            
            # Form clean address
            house_num = parse_str(osm_row.get('addr:housenumber'), '')
            street = parse_str(osm_row.get('addr:street'), '')
            if house_num and street:
                addr = f"{house_num} {street}"
            else:
                addr = parse_str(osm_row.get('name'), f"OSM Building {osm_id}")
                
            parent_unmatched = pd.Series({
                'geometry': osm_row.geometry,
                'type': 'building',
                'building': parse_str(osm_row.get('building'), 'yes'),
                'height': h_val,
                'height_p90': h_val,
                'height_max': h_val,
                'building:levels': levels_val,
                'name': parse_str(osm_row.get('name'), ''),
                'osm_id': osm_id,
                'BUILDING': -1,
                'lqs': 1.0,
                'q_coverage': 1.0,
                'q_canopy': 1.0,
                'tree_percentage': 0.0,
                'is_fallback': False,
                'address': addr,
                'bldg_use': osm_bldg_use,
                'max_floors': levels_val,
                'total_internal_sqft': round(osm_row.geometry.area * levels_val * 10.76391, 2),
                'clean_surface_area': round(osm_row.geometry.area * 2.0 + osm_row.geometry.boundary.length * h_val, 2),
                'clean_volume_total': round(osm_row.geometry.area * h_val, 2),
                'clean_area': round(osm_row.geometry.area, 2),
                'num_floors': levels_val,
                'internal_area_sqft': round(osm_row.geometry.area * levels_val * 10.76391, 2),
                'clean_volume': round(osm_row.geometry.area * h_val, 2)
            })
            blended_features.append(parent_unmatched)
            
    if blended_features:
        blended_df = pd.DataFrame(blended_features)
        blended_gdf = gpd.GeoDataFrame(blended_df, crs=utm_crs)
        return blended_gdf.to_crs("EPSG:4326")
    else:
        return gpd.GeoDataFrame(columns=s3db_parents.columns, crs="EPSG:4326")



class StepTimer:
    def __init__(self, name, timer_dict, num_buildings=None):
        self.name = name
        self.timer_dict = timer_dict
        self.num_buildings = num_buildings
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.timer_dict[self.name] = {
            'duration': duration,
            'num_buildings': self.num_buildings
        }
        print(f"⏱️  [{self.name}] completed in {duration:.2f} seconds.")




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


def fix_geometry(geom):
    """Try to fix invalid geometries or self-intersections."""
    if not geom.is_valid:
        return geom.buffer(0)
    return geom


def seal_polygon_holes(geom, max_hole_area=200.0):
    """
    Seals internal holes/voids in a polygon or multipolygon.
    Only fills holes that are smaller than max_hole_area to preserve large courtyards.
    """
    from shapely.geometry import Polygon, MultiPolygon
    if geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        interiors_to_keep = []
        for interior in geom.interiors:
            hole_poly = Polygon(interior)
            if hole_poly.area >= max_hole_area:
                interiors_to_keep.append(interior)
        return Polygon(geom.exterior, interiors_to_keep)
    elif isinstance(geom, MultiPolygon):
        parts = [seal_polygon_holes(p, max_hole_area) for p in geom.geoms]
        return MultiPolygon(parts)
    return geom


def clean_building_method_d(bldg, utm_crs, resolution=1.0, tolerance=0.3, max_hole_area=200.0,
                             height_round_step=3.0):
    """
    Cleans a single building's segments using Method D (Hole-Healing & Re-vectorization).

    Best-performing configuration (tested in scratch/method_d_healing.py):
      resolution=1m, height_round_step=3m (vertical rounding to merge HVAC/roof levels).

    Also computes per-segment and building-level metrics:
      - num_floors, internal_area_m2, internal_area_sqft
      - clean_area, clean_volume, clean_surface_area, total_internal_sqft, max_floors
      - bldg_use (Residential / Commercial / Other) derived from Overture metadata
    """
    import numpy as np
    import scipy.ndimage as ndimage
    import shapely
    import shapely.geometry
    from rasterio.features import rasterize, shapes
    from rasterio.transform import from_bounds
    import topojson as tp
    import pandas as _pd

    # 1. Project to UTM CRS
    bldg_utm = bldg.to_crs(utm_crs)
    bldg_utm.geometry = bldg_utm.geometry.apply(fix_geometry)

    # ── Building-type classification for floor height ──────────────────────────
    bldg_class   = bldg['class'].iloc[0]   if 'class'   in bldg.columns and not _pd.isna(bldg['class'].iloc[0])   else ''
    bldg_subtype = bldg['subtype'].iloc[0] if 'subtype' in bldg.columns and not _pd.isna(bldg['subtype'].iloc[0]) else ''
    bldg_names   = bldg['names'].iloc[0]   if 'names'   in bldg.columns and not _pd.isna(bldg['names'].iloc[0])   else ''

    bldg_class_str   = str(bldg_class).lower()
    bldg_subtype_str = str(bldg_subtype).lower()
    bldg_name_str    = str(bldg_names).lower()

    res_indicators = {'residential', 'apartments', 'detached', 'house', 'duplex'}
    com_indicators = {'commercial', 'retail', 'office', 'school', 'university', 'college',
                      'library', 'hospital', 'medical', 'civic', 'religious', 'church', 'hall'}

    is_residential = any(ind in bldg_class_str or ind in bldg_subtype_str or ind in bldg_name_str
                         for ind in res_indicators)
    is_commercial  = any(ind in bldg_class_str or ind in bldg_subtype_str or ind in bldg_name_str
                         for ind in com_indicators)

    # Default to commercial (McMaster campus context)
    if not is_residential and not is_commercial:
        is_commercial = True

    if is_residential:
        floor_height = 3.0
        bldg_use = "Residential"
    elif is_commercial:
        floor_height = 4.0
        bldg_use = "Commercial"
    else:
        floor_height = 3.5
        bldg_use = "Other/Default"

    # ── Height rounding (merges HVAC / minor roof components) ─────────────────
    if height_round_step is not None and height_round_step > 0.0:
        bldg_utm['height_p90'] = bldg_utm['height_p90'].apply(
            lambda h: max(height_round_step, round(float(h) / height_round_step) * height_round_step)
        )

    # Compute footprint union & seal holes
    footprint_union  = bldg_utm.geometry.union_all() if hasattr(bldg_utm.geometry, 'union_all') else bldg_utm.geometry.unary_union
    sealed_footprint = seal_polygon_holes(footprint_union, max_hole_area=max_hole_area)
    sealed_footprint = fix_geometry(sealed_footprint)

    if sealed_footprint.is_empty:
        return gpd.GeoDataFrame(columns=bldg.columns, crs=utm_crs)

    # 2. Setup Grid Bounds
    bounds = sealed_footprint.bounds  # minx, miny, maxx, maxy
    pad    = resolution * 2.0
    minx, miny, maxx, maxy = bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad

    nx = int(np.ceil((maxx - minx) / resolution))
    ny = int(np.ceil((maxy - miny) / resolution))

    # Safe boundary checks
    if nx <= 0 or ny <= 0 or (nx * ny) > 2_000_000:
        return clean_building_fallback(bldg, utm_crs)

    transform = from_bounds(minx, miny, maxx, maxy, nx, ny)

    # 3. Sort ascending so higher segments overwrite lower ones when rasterized
    bldg_sorted = bldg_utm.sort_values(by='height_p90', ascending=True)
    grid = np.zeros((ny, nx), dtype=np.float32)

    for _, row in bldg_sorted.iterrows():
        geom   = row.geometry
        height = float(row['height_p90'])
        rasterize([(geom, height)], out_shape=(ny, nx), transform=transform,
                  out=grid, fill=0, all_touched=True)

    footprint_mask = rasterize(
        [(sealed_footprint, 1)], out_shape=(ny, nx), transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    # 4. Nearest-Neighbor Gap Filling
    valid_mask = (grid > 0.0)
    if np.any(valid_mask):
        _, indices  = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
        grid_healed = grid[indices[0], indices[1]]
        grid_healed[~footprint_mask] = 0.0
    else:
        grid_healed = grid

    # 5. Re-vectorization
    mask_to_vector = (grid_healed > 0.0)
    shapes_gen     = shapes(grid_healed.astype(np.float32), mask=mask_to_vector, transform=transform)

    polys   = []
    heights = []
    for geom_dict, val in shapes_gen:
        poly = shapely.geometry.shape(geom_dict)
        if not poly.is_empty and poly.area > 0.01:
            polys.append(poly)
            heights.append(float(val))

    if not polys:
        return gpd.GeoDataFrame(columns=bldg.columns, crs=utm_crs)

    gdf_healed = gpd.GeoDataFrame({'height_p90': heights, 'geometry': polys}, crs=utm_crs)
    gdf_healed = gdf_healed.dissolve(by='height_p90').reset_index()
    gdf_healed.geometry = gdf_healed.geometry.apply(fix_geometry)

    # 6. Simplify topology
    if len(gdf_healed) > 1:
        gdf_healed.geometry = gdf_healed.geometry.buffer(0.02, join_style=2).apply(fix_geometry)
        try:
            topo             = tp.Topology(gdf_healed, prequantize=2000)
            topo_simplified  = topo.toposimplify(epsilon=tolerance)
            gdf_simplified   = topo_simplified.to_gdf()
        except Exception:
            gdf_simplified           = gdf_healed.copy()
            gdf_simplified.geometry  = gdf_simplified.geometry.simplify(tolerance, preserve_topology=True)
    else:
        gdf_simplified          = gdf_healed.copy()
        gdf_simplified.geometry = gdf_simplified.geometry.simplify(tolerance, preserve_topology=True)

    gdf_simplified.geometry = gdf_simplified.geometry.apply(fix_geometry)

    # 7. Per-segment metrics: area, volume, floors, internal sqft
    gdf_simplified = gdf_simplified.sort_values(by='height_p90', ascending=False)
    gdf_simplified['clean_area']         = gdf_simplified.geometry.area.round(2)
    gdf_simplified['clean_volume']       = (gdf_simplified['clean_area'] * gdf_simplified['height_p90']).round(2)
    gdf_simplified['num_floors']         = gdf_simplified['height_p90'].apply(
        lambda h: max(1, int(round(h / floor_height)))
    )
    gdf_simplified['internal_area_m2']   = (gdf_simplified['clean_area'] * gdf_simplified['num_floors']).round(2)
    gdf_simplified['internal_area_sqft'] = (gdf_simplified['internal_area_m2'] * 10.76391).round(2)

    total_volume        = gdf_simplified['clean_volume'].sum()
    total_roof_area     = gdf_simplified['clean_area'].sum()
    total_internal_sqft = gdf_simplified['internal_area_sqft'].sum()
    max_floors          = int(gdf_simplified['num_floors'].max())

    # Wall & sealed surface area
    simplified_footprint = gdf_simplified.geometry.union_all()
    footprint_area       = simplified_footprint.area

    total_wall_area = 0.0
    for _, row in gdf_simplified.iterrows():
        total_wall_area += row.geometry.boundary.length * row['height_p90']

    import shapely as _shapely
    shared_wall_reduction = 0.0
    sindex = gdf_simplified.sindex
    for i in range(len(gdf_simplified)):
        seg_i            = gdf_simplified.iloc[i]
        possible_neighbors = list(sindex.intersection(seg_i.geometry.bounds))
        for j_idx in possible_neighbors:
            if j_idx <= i:
                continue
            seg_j       = gdf_simplified.iloc[j_idx]
            shared_edge = _shapely.intersection(
                seg_i.geometry.boundary, seg_j.geometry.boundary, grid_size=0.1
            )
            if not shared_edge.is_empty and shared_edge.length > 0:
                min_h = min(seg_i['height_p90'], seg_j['height_p90'])
                shared_wall_reduction += 2.0 * (shared_edge.length * min_h)

    final_wall_area  = total_wall_area - shared_wall_reduction
    total_sealed_sa  = total_roof_area + footprint_area + final_wall_area

    # 8. Stamp building-level totals onto every row (for tooltip access in app.py)
    gdf_simplified['clean_surface_area']  = round(total_sealed_sa, 2)
    gdf_simplified['clean_volume_total']  = round(total_volume, 2)
    gdf_simplified['total_internal_sqft'] = round(total_internal_sqft, 2)
    gdf_simplified['max_floors']          = max_floors
    gdf_simplified['bldg_use']            = bldg_use
    gdf_simplified['floor_height']        = floor_height

    # 9. Map back original metadata attributes
    gdf_simplified['BUILDING'] = bldg['BUILDING'].iloc[0]
    if 'address' in bldg.columns:
        gdf_simplified['address'] = bldg['address'].iloc[0]
    else:
        gdf_simplified['address'] = f"Building {bldg['BUILDING'].iloc[0]}"
    if 'type' in bldg.columns:
        gdf_simplified['type'] = bldg['type'].iloc[0]
    else:
        gdf_simplified['type'] = 'Planar Roof Segment'

    if 'height_max' in bldg.columns:
        gdf_simplified['height_max'] = gdf_simplified['height_p90']

    # Carry over other attributes if present
    for col in ['SLOPE', 'ASPECT', 'AREA', 'VALUE']:
        if col in bldg.columns:
            gdf_simplified[col] = bldg[col].iloc[0]

    return gdf_simplified


def clean_building_fallback(bldg, utm_crs):
    """
    Fallback building cleaning logic.
    """
    bldg_utm = bldg.to_crs(utm_crs)
    bldg_utm.geometry = bldg_utm.geometry.apply(fix_geometry)
    bldg_utm.geometry = bldg_utm.geometry.simplify(1.0, preserve_topology=True)
    bldg_utm.geometry = bldg_utm.geometry.apply(fix_geometry)
    bldg_utm = bldg_utm.sort_values(by='height_p90', ascending=False)
    
    processed_footprints = None
    cleaned_features = []
    
    for idx, row in bldg_utm.iterrows():
        geom = row.geometry
        if processed_footprints is None:
            actual_geom = geom
            processed_footprints = geom
        else:
            try:
                actual_geom = geom.difference(processed_footprints.buffer(0.01))
                processed_footprints = unary_union([processed_footprints, geom]).buffer(0)
            except Exception:
                actual_geom = geom
        if not actual_geom.is_empty:
            actual_geom = fix_geometry(actual_geom)
            new_row = row.copy()
            new_row.geometry = actual_geom
            cleaned_features.append(new_row)
            
    if cleaned_features:
        return gpd.GeoDataFrame(cleaned_features, crs=utm_crs)
    else:
        return gpd.GeoDataFrame(columns=bldg.columns, crs=utm_crs)


def ensure_method_d_metrics(gdf, utm_crs, floor_height=4.0, bldg_use='Other/Default'):
    """
    Ensures that a GeoDataFrame has all Method D metrics calculated and populated.
    """
    if gdf.empty:
        return gdf
        
    # Check if core metrics are already in columns
    if 'clean_area' in gdf.columns and 'clean_surface_area' in gdf.columns:
        return gdf
        
    gdf = gdf.copy()
    gdf = gdf.sort_values(by='height_p90', ascending=False)
    
    gdf['clean_area'] = gdf.geometry.area.round(2)
    gdf['clean_volume'] = (gdf['clean_area'] * gdf['height_p90']).round(2)
    gdf['num_floors'] = gdf['height_p90'].apply(
        lambda h: max(1, int(round(h / floor_height)))
    )
    gdf['internal_area_m2'] = (gdf['clean_area'] * gdf['num_floors']).round(2)
    gdf['internal_area_sqft'] = (gdf['internal_area_m2'] * 10.76391).round(2)
    
    total_volume = gdf['clean_volume'].sum()
    total_roof_area = gdf['clean_area'].sum()
    total_internal_sqft = gdf['internal_area_sqft'].sum()
    max_floors = int(gdf['num_floors'].max())
    
    # Sealed surface area: roofs + footprint + walls
    footprint_area = gdf.geometry.union_all().area
    total_wall_area = sum(row.geometry.boundary.length * row['height_p90'] for _, row in gdf.iterrows())
    total_sealed_sa = total_roof_area + footprint_area + (total_wall_area * 0.8) # 20% shared walls reduction estimate
    
    gdf['clean_surface_area'] = round(total_sealed_sa, 2)
    gdf['clean_volume_total'] = round(total_volume, 2)
    gdf['total_internal_sqft'] = round(total_internal_sqft, 2)
    gdf['max_floors'] = max_floors
    gdf['bldg_use'] = bldg_use
    gdf['floor_height'] = floor_height
    
    return gdf


def clean_rooftops(input_path: Path, output_path: Path, utm_crs: str, footprints_path: Path, roughness_path: Path, cache_file: Path = None, raw_osm_path: Path = None, lqs_threshold: float = 0.70):
    """
    Groups raw rooftop segments by building, cleans them using Method D, calculates the
    LiDAR Quality Score (LQS), applies flat roof fallback for low quality, and exports S3DB data.
    """
    print(f"🧹 Cleaning and simplifying rooftops with Method D and calculating quality metrics...")
    gdf = gpd.read_file(input_path)
    
    original_crs = gdf.crs
    
    # Load original footprints for LQS calculation
    footprints_gdf = None
    if footprints_path.exists():
        footprints_gdf = gpd.read_file(footprints_path)
        footprints_gdf = footprints_gdf.to_crs(utm_crs)
    
    cleaned_parts = []
    parents_list = []
    parts_list = []
    
    building_ids = gdf['BUILDING'].unique()
    
    for b_id in building_ids:
        bldg = gdf[gdf['BUILDING'] == b_id].copy()
        
        # Get parent geometry
        parent_geom = None
        parent_row_data = {}
        if footprints_gdf is not None and not footprints_gdf.empty:
            if 'BUILDING' in footprints_gdf.columns:
                matched = footprints_gdf[footprints_gdf['BUILDING'] == b_id]
                if not matched.empty:
                    parent_geom = matched.geometry.iloc[0]
                    parent_row_data = matched.iloc[0].to_dict()
            if parent_geom is None and int(b_id) < len(footprints_gdf):
                parent_geom = footprints_gdf.geometry.iloc[int(b_id)]
                parent_row_data = footprints_gdf.iloc[int(b_id)].to_dict()
        
        try:
            # Clean using Method D
            cleaned_bldg = clean_building_method_d(bldg, utm_crs)
            
            # If cleaning yielded no geometries, fall back to simple subtract
            if cleaned_bldg.empty:
                cleaned_bldg = clean_building_fallback(bldg, utm_crs)
                
            # If still empty, use the parent footprint itself as fallback
            if cleaned_bldg.empty and parent_geom is not None:
                max_h = float(bldg['height_p90'].max()) if not bldg.empty else 10.0
                fallback_num_floors = max(1, int(round(max_h / 4.0)))
                fallback_dict = {
                    'height_p90': [max_h],
                    'height_max': [max_h],
                    'geometry': [parent_geom],
                    'clean_area': [parent_geom.area],
                    'clean_volume': [parent_geom.area * max_h],
                    'num_floors': [fallback_num_floors],
                    'internal_area_m2': [parent_geom.area * fallback_num_floors],
                    'internal_area_sqft': [parent_geom.area * fallback_num_floors * 10.76391],
                    'clean_surface_area': [parent_geom.area * 2.0 + parent_geom.boundary.length * max_h],
                    'clean_volume_total': [parent_geom.area * max_h],
                    'total_internal_sqft': [parent_geom.area * fallback_num_floors * 10.76391],
                    'max_floors': [fallback_num_floors],
                    'bldg_use': ['Other/Default'],
                    'floor_height': [4.0],
                    'BUILDING': [b_id]
                }
                cleaned_bldg = gpd.GeoDataFrame(fallback_dict, crs=utm_crs)
                
            # Calculate LiDAR Quality Score (LQS)
            lqs_dict = {
                'lqs': 1.0,
                'q_coverage': 1.0,
                'q_canopy': 1.0,
                'tree_percentage': 0.0
            }
            is_fallback = False
            
            if parent_geom is not None:
                method_d_union = cleaned_bldg.union_all() if not cleaned_bldg.empty else None
                lqs_dict = quality_metrics.calculate_lqs(parent_geom, method_d_union, roughness_path, utm_crs)
                
                # Check quality threshold
                if lqs_dict['lqs'] < lqs_threshold:
                    print(f"   ⚠️ Building {b_id} has LQS={lqs_dict['lqs']:.2f} (< {lqs_threshold:.2f}). Falling back to flat roof.")
                    is_fallback = True
                    max_h = float(bldg['height_p90'].max()) if not bldg.empty else 10.0
                    
                    fallback_dict = {
                        'height_p90': [max_h],
                        'height_max': [max_h],
                        'geometry': [parent_geom],
                        'clean_area': [parent_geom.area],
                        'clean_volume': [parent_geom.area * max_h],
                        'num_floors': [max(1, int(round(max_h / 4.0)))],
                        'internal_area_m2': [parent_geom.area * max(1, int(round(max_h / 4.0)))],
                        'internal_area_sqft': [parent_geom.area * max(1, int(round(max_h / 4.0))) * 10.76391],
                        'clean_surface_area': [parent_geom.area * 2.0 + parent_geom.boundary.length * max_h],
                        'clean_volume_total': [parent_geom.area * max_h],
                        'total_internal_sqft': [parent_geom.area * max(1, int(round(max_h / 4.0))) * 10.76391],
                        'max_floors': [max(1, int(round(max_h / 4.0)))],
                        'bldg_use': ['Flat Roof (Fallback)'],
                        'floor_height': [4.0],
                        'BUILDING': [b_id],
                        'address': [f"Building {b_id} (Fallback)"],
                        'type': ['Flat Roof (Tree Fallback)']
                    }
                    cleaned_bldg = gpd.GeoDataFrame(fallback_dict, crs=utm_crs)
            
            # Ensure all Method D columns exist in cleaned_bldg
            bldg_use_str = parent_row_data.get('building', 'Other/Default')
            if not isinstance(bldg_use_str, str) or pd.isna(bldg_use_str):
                bldg_use_str = 'Other/Default'
            cleaned_bldg = ensure_method_d_metrics(cleaned_bldg, utm_crs, bldg_use=bldg_use_str.title())
            
            # Format and save S3DB features
            if not cleaned_bldg.empty and parent_geom is not None:
                max_height = float(cleaned_bldg['height_p90'].max())
                max_floors = int(cleaned_bldg['num_floors'].max())
                total_int_sqft = float(cleaned_bldg['internal_area_sqft'].sum()) if 'internal_area_sqft' in cleaned_bldg.columns else 0.0
                bldg_use = cleaned_bldg['bldg_use'].iloc[0] if 'bldg_use' in cleaned_bldg.columns else 'Other/Default'
                address = cleaned_bldg['address'].iloc[0] if 'address' in cleaned_bldg.columns else f"Building {b_id}"
                
                clean_surface_area = float(cleaned_bldg['clean_surface_area'].iloc[0]) if 'clean_surface_area' in cleaned_bldg.columns else 0.0
                clean_volume_total = float(cleaned_bldg['clean_volume_total'].iloc[0]) if 'clean_volume_total' in cleaned_bldg.columns else 0.0
                
                # Create parent S3DB building feature
                parent_feat = {
                    'geometry': parent_geom,
                    'type': 'building',
                    'building': 'yes' if bldg_use == 'Other/Default' else bldg_use.lower(),
                    'height': max_height,
                    'height_p90': max_height,
                    'height_max': max_height,
                    'building:levels': max_floors,
                    'BUILDING': b_id,
                    'lqs': lqs_dict['lqs'],
                    'q_coverage': lqs_dict['q_coverage'],
                    'q_canopy': lqs_dict['q_canopy'],
                    'tree_percentage': lqs_dict['tree_percentage'],
                    'is_fallback': is_fallback,
                    'address': address,
                    'name': parent_row_data.get('names', parent_row_data.get('name', '')),
                    # Building Totals
                    'total_internal_sqft': round(total_int_sqft, 2),
                    'clean_surface_area': round(clean_surface_area, 2),
                    'clean_volume_total': round(clean_volume_total, 2),
                    'bldg_use': bldg_use,
                    'max_floors': max_floors,
                    # Segment/Parent overall metrics (for tooltip fallback)
                    'clean_area': round(parent_geom.area, 2),
                    'num_floors': max_floors,
                    'internal_area_sqft': round(total_int_sqft, 2),
                    'clean_volume': round(clean_volume_total, 2)
                }
                parents_list.append(parent_feat)
                
                # Create child building parts
                for _, row in cleaned_bldg.iterrows():
                    part_feat = {
                        'geometry': row.geometry,
                        'type': 'building_part',
                        'building:part': 'yes',
                        'height': row.height_p90,
                        'height_p90': row.height_p90,
                        'height_max': row.height_p90,
                        'min_height': 0.0,
                        'building:levels': row.num_floors,
                        'parent_id': b_id,
                        'BUILDING': b_id,
                        'is_fallback': is_fallback,
                        'address': address,
                        'lqs': lqs_dict['lqs'],
                        # Building Totals (propagated to parts)
                        'q_coverage': lqs_dict['q_coverage'],
                        'q_canopy': lqs_dict['q_canopy'],
                        'tree_percentage': lqs_dict['tree_percentage'],
                        'bldg_use': bldg_use,
                        'max_floors': max_floors,
                        'total_internal_sqft': round(total_int_sqft, 2),
                        'clean_surface_area': round(clean_surface_area, 2),
                        'clean_volume_total': round(clean_volume_total, 2),
                        # Segment-specific metrics
                        'clean_area': row.clean_area,
                        'num_floors': row.num_floors,
                        'internal_area_sqft': row.internal_area_sqft,
                        'clean_volume': row.clean_volume
                    }
                    parts_list.append(part_feat)
                    
            cleaned_parts.append(cleaned_bldg)
            
        except Exception as e:
            print(f"❌ Method D and Fallbacks failed for building {b_id}: {e}")
            import traceback
            traceback.print_exc()

    if cleaned_parts:
        # Save clean rooftops (re-assembled parts layer)
        clean_gdf = pd.concat(cleaned_parts, ignore_index=True)
        clean_gdf = gpd.GeoDataFrame(clean_gdf, crs=utm_crs)
        clean_gdf = clean_gdf.to_crs(original_crs)
        clean_gdf.to_file(output_path, driver="GeoJSON")
        print(f"✅ Method D Cleaned block model saved to: {output_path.name}")
        
        # Save S3DB combined parents and parts model
        s3db_parents = gpd.GeoDataFrame(parents_list, crs=utm_crs).to_crs(original_crs)
        s3db_parts = gpd.GeoDataFrame(parts_list, crs=utm_crs).to_crs(original_crs)
        
        s3db_combined = pd.concat([s3db_parents, s3db_parts], ignore_index=True)
        s3db_combined = gpd.GeoDataFrame(s3db_combined, crs=original_crs)
        
        s3db_output_path = output_path.parent / output_path.name.replace("clean_3d.geojson", "s3db_3d.geojson")
        s3db_combined.to_file(s3db_output_path, driver="GeoJSON")
        print(f"✅ S3DB Simple 3D Buildings model saved to: {s3db_output_path.name}")
        
        # Fetch OSM buildings for blending
        # Define search box for overpass query based on footprints_gdf bounds in WGS84
        if footprints_gdf is not None:
            wgs_footprints = footprints_gdf.to_crs("EPSG:4326")
            bounds = wgs_footprints.total_bounds # xmin, ymin, xmax, ymax
            # Add small padding (approx 50m)
            bbox = [bounds[0] - 0.0005, bounds[1] - 0.0005, bounds[2] + 0.0005, bounds[3] + 0.0005]
            
            osm_gdf = fetch_osm_buildings(bbox, cache_file, raw_osm_path)
            
            # Blend with OSM
            blended_gdf = blend_with_osm(s3db_parents, s3db_parts, osm_gdf, utm_crs)
            blended_output_path = output_path.parent / output_path.name.replace("clean_3d.geojson", "blended_3d.geojson")
            blended_gdf.to_file(blended_output_path, driver="GeoJSON")
            print(f"✅ Blended OSM 3D dataset saved to: {blended_output_path.name}")
    else:
        print("⚠️ No features survived cleaning.")


def main():
    pipeline_times = {}
    # Define the Province and City you are processing here!
    province_name = "Ontario"
    city_name = "McMaster" 
    
    # Bounding Box (Lat/Lon) [min_lon, min_lat, max_lon, max_lat]
    # Default McMaster bbox (will be overridden if LiDAR file is found)
    bbox = [-79.925, 43.255, -79.910, 43.268]

    # Parse command line flags
    force = False
    if "--force" in sys.argv:
        force = True

    # 1. Setup Province-based Project Structure
    province_dir = Path(f"./data/{province_name}")
    footprints_dir = province_dir / "footprints"
    city_raw_dir = province_dir / city_name / "raw"
    city_processed_dir = province_dir / city_name / "processed"
    
    # Create directories early so we can check for files
    footprints_dir.mkdir(parents=True, exist_ok=True)
    city_raw_dir.mkdir(parents=True, exist_ok=True)
    city_processed_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = city_processed_dir / "pipeline_hashes.json"

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
        
        # Check if we can load from cache
        lidar_metadata_loaded = False
        lidar_file_hash = pipeline_cache.get_file_hash(lidar_file)
        
        if not force and cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                meta = cache_data.get("lidar_metadata", {})
                if meta and meta.get("file_hash") == lidar_file_hash:
                    num_points = meta.get("num_points", 0)
                    bbox = meta.get("bbox")
                    lidar_crs = meta.get("lidar_crs")
                    lidar_metadata_loaded = True
                    print(f"ℹ️ Loaded LiDAR metadata from cache. Points: {num_points:,}")
                    print(f"🎯 Study area synced to LiDAR tile (cached): {bbox}")
            except Exception as e:
                print(f"⚠️ Failed to load LiDAR metadata from cache: {e}. Reading LiDAR file...")
                
        if not lidar_metadata_loaded:
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
                
                # Save metadata to cache
                try:
                    import json
                    cache_data = {}
                    if cache_file.exists():
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                    
                    cache_data["lidar_metadata"] = {
                        "file_path": str(lidar_file.resolve()),
                        "file_hash": lidar_file_hash,
                        "num_points": num_points,
                        "bbox": bbox,
                        "lidar_crs": lidar_crs
                    }
                    
                    temp_file = cache_file.with_suffix('.tmp')
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    import os
                    os.replace(temp_file, cache_file)
                    print(f"✅ Cached LiDAR metadata to: {cache_file.name}")
                except Exception as e:
                    print(f"⚠️ Failed to cache LiDAR metadata: {e}")
    
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

    # =========================================================================
    # Stage 1: Raster Preprocessing
    # =========================================================================
    stage1_outputs = [
        city_processed_dir / f"{city_name.lower()}_ndsm_output.tif",
        city_processed_dir / f"{city_name.lower()}_smoothed_ndsm.tif",
        city_processed_dir / f"{city_name.lower()}_roughness_output.tif"
    ]
    stage1_inputs = [dsm_file, dtm_file]
    stage1_config = {
        "filter_size": 11,
        "normal_diff_threshold": 18.0
    }
    
    run_stage1 = pipeline_cache.should_run_stage(
        stage_name="Raster Preprocessing",
        input_files=stage1_inputs,
        config=stage1_config,
        output_files=stage1_outputs,
        cache_file=cache_file,
        force=force
    )
    
    if run_stage1:
        # 3. Read the Lidar Raster Data into Memory
        with StepTimer("Raster Load & nDSM", pipeline_times):
            print(f"\nLoading DSM and DTM into memory for {city_name}...")
            dsm = wbe.read_raster(str(dsm_file))
            dtm = wbe.read_raster(str(dtm_file))
            
            # 4. Calculate Normalized Digital Surface Model (nDSM)
            print("Calculating nDSM...")
            ndsm = dsm - dtm
            wbe.write_raster(ndsm, str(stage1_outputs[0])) 

        # 4.5. Feature Preserving Smoothing
        with StepTimer("Feature Preserving Smoothing", pipeline_times):
            print("Applying Feature Preserving Smoothing to nDSM...")
            smoothed_ndsm = wbe.feature_preserving_smoothing(
                ndsm, 
                filter_size=stage1_config["filter_size"], 
                normal_diff_threshold=stage1_config["normal_diff_threshold"]
            )
            wbe.write_raster(smoothed_ndsm, str(stage1_outputs[1]))
        
        # 5. Surface Roughness Filter (Removing Trees)
        with StepTimer("Surface Ruggedness Filter (Trees)", pipeline_times):
            print("Applying Surface Ruggedness Filter to identify trees...")
            roughness = wbe.ruggedness_index(smoothed_ndsm)
            wbe.write_raster(roughness, str(stage1_outputs[2]))
            
        pipeline_cache.update_stage_cache(
            stage_name="Raster Preprocessing",
            input_files=stage1_inputs,
            config=stage1_config,
            cache_file=cache_file
        )
    else:
        print("⏭️  Skipping Stage 1: Raster Preprocessing (results reused from cache).")

    # =========================================================================
    # Stage 2: Native Footprint Extraction
    # =========================================================================
    stage2_output = city_processed_dir / f"{city_name.lower()}_lidar_buildings_3d.geojson"
    stage2_inputs = [
        city_processed_dir / f"{city_name.lower()}_ndsm_output.tif",
        city_processed_dir / f"{city_name.lower()}_roughness_output.tif"
    ]
    stage2_config = {
        "utm_crs": lidar_crs,
        "min_area": 25.0
    }
    
    run_stage2 = pipeline_cache.should_run_stage(
        stage_name="Native LiDAR Footprint Extraction",
        input_files=stage2_inputs,
        config=stage2_config,
        output_files=[stage2_output],
        cache_file=cache_file,
        force=force
    )
    
    if run_stage2:
        print(f"\nExtracting Native Footprints directly from LiDAR...")
        with StepTimer("Native LiDAR Footprint Extraction", pipeline_times) as timer:
            try:
                import rasterio
                import numpy as np
                import shapely
                
                ndsm_path = str(stage2_inputs[0])
                rough_path = str(stage2_inputs[1])
                
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
                lidar_gdf = lidar_gdf[lidar_gdf.to_crs(lidar_crs).geometry.area >= stage2_config["min_area"]]
                
                lidar_gdf['height_p90'] = lidar_gdf['VALUE']
                lidar_gdf['height_max'] = lidar_gdf['VALUE']
                lidar_gdf['address'] = 'LiDAR Auto-Extracted'
                lidar_gdf['type'] = 'Multi-Tiered Polygon'
                
                lidar_gdf.geometry = shapely.force_2d(lidar_gdf.geometry)
                
                print(f"Saving final LiDAR dataset to: {stage2_output.name}...")
                lidar_gdf.to_file(stage2_output, driver="GeoJSON")
                print(f"✅ Extracted {len(lidar_gdf)} native LiDAR sub-polygons.")
                timer.num_buildings = len(lidar_gdf)
                
                # Clean up temporary shapefile created by Whitebox internally
                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                    p = city_processed_dir / f"{city_name.lower()}_wb_extracted{ext}"
                    if p.exists():
                        p.unlink()
                if terraced_path.exists():
                    terraced_path.unlink()
                
                pipeline_cache.update_stage_cache(
                    stage_name="Native LiDAR Footprint Extraction",
                    input_files=stage2_inputs,
                    config=stage2_config,
                    cache_file=cache_file
                )
            except Exception as e:
                print(f"❌ Failed to extract LiDAR footprints: {e}")
    else:
        print("⏭️  Skipping Stage 2: Native LiDAR Footprint Extraction (results reused from cache).")

    # =========================================================================
    # Stage 3 & 4: Load Footprints and Zonal Statistics
    # =========================================================================
    print(f"\nLoading Building Footprints...")
    datasets = {}
    
    with StepTimer("Load Vector Footprints", pipeline_times) as timer:
        # Load StatCan Footprints
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
                    if datasets["statcan"].crs is None:
                        datasets["statcan"].set_crs(native_crs, inplace=True, allow_override=True)
                    print(f"✅ Loaded {len(datasets['statcan'])} StatCan buildings.")
        except Exception as e:
            print(f"❌ Failed to load StatCan data: {e}")
            
        # Load Overture Footprints
        overture_path = city_raw_dir / "overture_footprints.geojson"
        if overture_path.exists():
            overture_gdf = gpd.read_file(overture_path)
            datasets["overture"] = overture_gdf
            print(f"✅ Loaded {len(overture_gdf)} Overture buildings.")
            
        timer.num_buildings = len(datasets.get("statcan", [])) + len(datasets.get("overture", []))

    # Zonal Stats
    from rasterio.mask import mask
    import numpy as np
    from tqdm import tqdm
    import shapely
    
    ndsm_path = str(city_processed_dir / f"{city_name.lower()}_ndsm_output.tif")
    
    for dataset_name, buildings_gdf in datasets.items():
        step_name = f"Zonal Stats ({dataset_name.upper()})"
        
        # Determine the input files for this dataset's zonal stats
        if dataset_name == "statcan":
            inp_files = list(footprints_dir.glob("*.shp")) + list(footprints_dir.glob("*.gpkg"))
        else:
            inp_files = [city_raw_dir / "overture_footprints.geojson"]
            
        inp_files.append(Path(ndsm_path))
        output_file = city_processed_dir / f"{city_name.lower()}_{dataset_name}_buildings_3d.geojson"
        zonal_config = {
            "bbox": bbox,
            "utm_crs": lidar_crs
        }
        
        run_zonal = pipeline_cache.should_run_stage(
            stage_name=step_name,
            input_files=inp_files,
            config=zonal_config,
            output_files=[output_file],
            cache_file=cache_file,
            force=force
        )
        
        if run_zonal:
            with StepTimer(step_name, pipeline_times, len(buildings_gdf)):
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
                
                # Filter out buildings with 0 or negative heights
                valid_buildings = buildings_gdf[buildings_gdf['height_p90'] >= 2.0].copy()
                print(f"Extracted heights for {len(valid_buildings)} valid {dataset_name} buildings.")
                
                # Export Final Output
                valid_buildings = valid_buildings.to_crs("EPSG:4326")
                valid_buildings.geometry = shapely.force_2d(valid_buildings.geometry)
                
                print(f"Saving final 3D dataset to: {output_file.name}...")
                valid_buildings.to_file(output_file, driver="GeoJSON")
                
            pipeline_cache.update_stage_cache(
                stage_name=step_name,
                input_files=inp_files,
                config=zonal_config,
                cache_file=cache_file
            )
        else:
            print(f"⏭️  Skipping {step_name} (results reused from cache).")

    # =========================================================================
    # Stage 5: LiDAR Rooftop Analysis (Phase 3: High-Detail Pass)
    # =========================================================================
    rooftops_output = city_processed_dir / f"{city_name.lower()}_lidar_rooftops_3d.geojson"
    rooftop_footprints_shp = city_processed_dir / f"{city_name.lower()}_rooftop_footprints.shp"
    
    if not datasets:
        print("❌ No building footprint datasets were loaded! Cannot proceed with rooftop analysis.")
        return
        
    if lidar_file:
        # Choose footprints: prefer Overture (cleaner), fallback to StatCan
        footprint_source = None
        if "overture" in datasets:
            footprint_source = "overture"
        elif "statcan" in datasets:
            footprint_source = "statcan"
        
        if footprint_source is None:
            print("⚠️  No footprint datasets available for rooftop analysis. Skipping Phase 3.")
        else:
            rooftop_inputs = [
                lidar_file,
                city_processed_dir / f"{city_name.lower()}_{footprint_source}_buildings_3d.geojson"
            ]
            rooftop_config = {
                "num_iterations": 50,
                "utm_crs": lidar_crs,
                "footprint_source": footprint_source
            }
            rooftop_outputs = [
                rooftops_output,
                rooftop_footprints_shp
            ]
            
            run_rooftop_analysis = pipeline_cache.should_run_stage(
                stage_name="LiDAR Rooftop Analysis (Whitebox)",
                input_files=rooftop_inputs,
                config=rooftop_config,
                output_files=rooftop_outputs,
                cache_file=cache_file,
                force=force
            )
            
            if run_rooftop_analysis:
                print("\n🏗️ Starting Phase 3: High-Detail LiDAR Rooftop Analysis...")
                try:
                    # Load the point cloud for processing
                    print("Reading point cloud data...")
                    lidar_data = wbe.read_lidar(str(lidar_file))
                    print(f"   Loaded {lidar_data.header.number_of_points:,} points")
                    
                    if lidar_data.header.number_of_points == 0:
                        print("⚠️  Cannot run rooftop analysis — 0 points in LiDAR file.")
                    else:
                        print(f"Running lidar_rooftop_analysis using {footprint_source.upper()} footprints...")
                        
                        # Load footprints from the zonal stats output
                        footprints_gdf = gpd.read_file(rooftop_inputs[1])
                        # Reproject footprints to match LiDAR CRS
                        footprints_gdf = footprints_gdf.to_crs(lidar_crs)
                        
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
                            # Save to permanent shapefile
                            footprints_gdf.to_file(rooftop_footprints_shp)
                            footprints_vec = wbe.read_vector(str(rooftop_footprints_shp))
                            
                            # Run the rooftop analysis
                            with StepTimer("LiDAR Rooftop Analysis (Whitebox)", pipeline_times, len(footprints_gdf)):
                                rooftops = wbe.lidar_rooftop_analysis(
                                    lidar_inputs=[lidar_data],
                                    building_footprints=footprints_vec,
                                    num_iterations=rooftop_config["num_iterations"]
                                )
                                
                                rooftops_shp = city_processed_dir / f"{city_name.lower()}_lidar_rooftops.shp"
                                wbe.write_vector(rooftops, str(rooftops_shp))
                                
                                # Convert results to 4326 GeoJSON for visualization
                                rooftops_gdf = gpd.read_file(rooftops_shp)
                                rooftops_gdf = rooftops_gdf.set_crs(lidar_crs, allow_override=True)
                                
                                # Filter out very small artifacts
                                rooftops_gdf = rooftops_gdf[rooftops_gdf.geometry.area >= 15.0].copy()
                                
                                # Compute actual height above ground (MAX_ELEV - DTM)
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
                                rooftops_gdf.to_file(rooftops_output, driver="GeoJSON")
                                print(f"✅ High-detail rooftops saved to: {rooftops_output.name}")
                                
                                # Clean up temporary shapefile created by Whitebox internally
                                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                                    p = city_processed_dir / f"{city_name.lower()}_lidar_rooftops{ext}"
                                    if p.exists():
                                        p.unlink()
                                        
                            pipeline_cache.update_stage_cache(
                                stage_name="LiDAR Rooftop Analysis (Whitebox)",
                                input_files=rooftop_inputs,
                                config=rooftop_config,
                                cache_file=cache_file
                            )
                except Exception as e:
                    print(f"❌ LiDAR Rooftop Analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⏭️  Skipping Stage 5: LiDAR Rooftop Analysis (results reused from cache).")

            # =========================================================================
            # Stage 6: Method D Cleaning & OSM Blending
            # =========================================================================
            clean_output = city_processed_dir / f"{city_name.lower()}_lidar_rooftops_clean_3d.geojson"
            s3db_output = city_processed_dir / f"{city_name.lower()}_lidar_rooftops_s3db_3d.geojson"
            blended_output = city_processed_dir / f"{city_name.lower()}_lidar_rooftops_blended_3d.geojson"
            
            stage6_inputs = [
                rooftops_output,
                city_processed_dir / f"{city_name.lower()}_roughness_output.tif",
                rooftop_footprints_shp
            ]
            stage6_config = {
                "utm_crs": lidar_crs,
                "lqs_threshold": 0.70,
                "bbox": bbox
            }
            stage6_outputs = [
                clean_output,
                s3db_output,
                blended_output
            ]
            
            run_stage6 = pipeline_cache.should_run_stage(
                stage_name="Method D Cleaning & OSM Blending",
                input_files=stage6_inputs,
                config=stage6_config,
                output_files=stage6_outputs,
                cache_file=cache_file,
                force=force
            )
            
            if run_stage6:
                # Phase 3b: Cleaned Block Model
                with StepTimer("Method D Cleaning & Simplification", pipeline_times):
                    clean_rooftops(
                        input_path=rooftops_output,
                        output_path=clean_output,
                        utm_crs=lidar_crs,
                        footprints_path=rooftop_footprints_shp,
                        roughness_path=city_processed_dir / f"{city_name.lower()}_roughness_output.tif",
                        cache_file=cache_file,
                        raw_osm_path=city_processed_dir / "osm_buildings_raw.geojson",
                        lqs_threshold=stage6_config["lqs_threshold"]
                    )
                
                pipeline_cache.update_stage_cache(
                    stage_name="Method D Cleaning & OSM Blending",
                    input_files=stage6_inputs,
                    config=stage6_config,
                    cache_file=cache_file
                )
            else:
                print("⏭️  Skipping Stage 6: Method D Cleaning & OSM Blending (results reused from cache).")

    print("\n🎉 Pipeline Complete! Comparative datasets processed.")
    
    # Print beautiful performance report
    if pipeline_times:
        print("\n" + "=" * 80)
        print("⏱️  PIPELINE PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Step Name':<42} | {'Time (s)':<10} | {'Bldgs':<8} | {'Avg/Bldg':<12}")
        print("-" * 80)
        for name, info in pipeline_times.items():
            dur = info['duration']
            nb = info['num_buildings']
            if nb is not None and nb > 0:
                avg = f"{dur / nb * 1000:.1f} ms"
                nb_str = f"{nb}"
            else:
                avg = "N/A"
                nb_str = "N/A"
            print(f"{name:<42} | {dur:<10.2f} | {nb_str:<8} | {avg:<12}")
        print("=" * 80)

if __name__ == "__main__":
    main()
