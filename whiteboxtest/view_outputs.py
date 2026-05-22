import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd

def main():
    # Set up paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    
    dem_path = os.path.join(output_dir, 'DEM.tif')
    buildings_raster_path = os.path.join(output_dir, 'buildings.tif')
    shapefile_path = os.path.join(output_dir, 'building_footprints.shp')
    
    # Check if files exist
    for path in [dem_path, buildings_raster_path, shapefile_path]:
        if not os.path.exists(path):
            print(f"Error: Could not find required output file: {path}")
            print("Please run whiteboxtest/wbwbuildingmapping.py first to generate the outputs.")
            return

    print("Loading data...")
    # Load rasters using rasterio
    dem_src = rasterio.open(dem_path)
    buildings_src = rasterio.open(buildings_raster_path)
    
    # Load shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)

    print("Generating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    # 1. Plot DEM
    axes[0].set_title("1. Digital Elevation Model (DEM)")
    show(dem_src, ax=axes[0], cmap='terrain')
    
    # 2. Plot Buildings Raster
    axes[1].set_title("2. Buildings Raster (Boolean)")
    show(buildings_src, ax=axes[1], cmap='binary')
    
    # 3. Plot Vector Footprints Overlaid on DEM
    axes[2].set_title("3. Building Footprints (Vector Overlay)")
    show(dem_src, ax=axes[2], cmap='gist_earth', alpha=0.6)
    if not gdf.empty:
        gdf.plot(ax=axes[2], facecolor='red', edgecolor='darkred', alpha=0.7)
    
    # Beautify layout
    plt.tight_layout()
    
    # Save the output image
    viz_png_path = os.path.join(output_dir, 'visualization.png')
    plt.savefig(viz_png_path, dpi=300)
    print(f"Visualization saved to: {viz_png_path}")
    
    # Try to show the interactive plot window
    try:
        print("Opening interactive window... Close window to exit script.")
        plt.show()
    except Exception as e:
        print(f"Could not open GUI window: {e}")
        print("You can view the saved image at: whiteboxtest/output/visualization.png")

if __name__ == "__main__":
    main()
