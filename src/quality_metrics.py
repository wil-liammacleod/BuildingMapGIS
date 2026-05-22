import numpy as np
import rasterio
from rasterio.mask import mask
import shapely

def calculate_lqs(overture_footprint, method_d_union, roughness_raster_path, utm_crs):
    """
    Calculates the LiDAR Quality Score (LQS) for a building based on:
    1. Footprint Coverage Ratio (Q_coverage): how much of the original footprint is represented by Method D parts.
    2. Tree Canopy Overlap (Q_canopy): fraction of the building footprint free of high roughness (trees).
    
    Parameters:
        overture_footprint (shapely.geometry.base.BaseGeometry): Footprint in UTM CRS.
        method_d_union (shapely.geometry.base.BaseGeometry): Union of all Method D parts in UTM CRS.
        roughness_raster_path (str or Path): Path to the ruggedness index TIFF file.
        utm_crs (str): The UTM projection EPSG code (e.g. 'EPSG:2958').
        
    Returns:
        dict: Containing 'lqs', 'q_coverage', 'q_canopy', and 'tree_percentage'.
    """
    # Fallback default values
    q_coverage = 0.0
    q_canopy = 1.0
    tree_percentage = 0.0
    
    # 1. Calculate Footprint Coverage Ratio
    if overture_footprint and not overture_footprint.is_empty:
        orig_area = overture_footprint.area
        if orig_area > 0:
            if method_d_union and not method_d_union.is_empty:
                try:
                    intersection = overture_footprint.intersection(method_d_union)
                    q_coverage = min(1.0, intersection.area / orig_area)
                except Exception as e:
                    print(f"⚠️ Error computing geometry intersection for coverage: {e}")
                    # Fallback to simple area ratio if intersection fails
                    q_coverage = min(1.0, method_d_union.area / orig_area)
            else:
                q_coverage = 0.0
    
    # 2. Calculate Tree Canopy Overlap from Roughness Raster
    if overture_footprint and not overture_footprint.is_empty and roughness_raster_path.exists():
        try:
            with rasterio.open(roughness_raster_path) as src:
                # Mask raster with the footprint polygon
                # crop=True crops the raster dataset to the footprint's bounding box
                out_image, _ = mask(src, [overture_footprint], crop=True, nodata=-9999)
                valid_pixels = out_image[out_image != -9999]
                
                if len(valid_pixels) > 0:
                    # Trees/vegetation typically have a roughness index >= 1.0
                    tree_pixels = np.sum(valid_pixels >= 1.0)
                    tree_percentage = (tree_pixels / len(valid_pixels)) * 100.0
                    q_canopy = max(0.0, 1.0 - (tree_pixels / len(valid_pixels)))
                else:
                    q_canopy = 1.0
                    tree_percentage = 0.0
        except Exception as e:
            print(f"⚠️ Error reading roughness raster for quality calculation: {e}")
            q_canopy = 1.0
            tree_percentage = 0.0
            
    # 3. Combine Metrics into LQS
    lqs = 0.5 * q_coverage + 0.5 * q_canopy
    
    return {
        'lqs': round(lqs, 3),
        'q_coverage': round(q_coverage, 3),
        'q_canopy': round(q_canopy, 3),
        'tree_percentage': round(tree_percentage, 1)
    }
