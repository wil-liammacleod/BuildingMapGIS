import geopandas as gpd
from shapely.ops import unary_union

path = "/Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/data/Ontario/McMaster/processed/mcmaster_lidar_rooftops_3d.geojson"
gdf = gpd.read_file(path)
bldg = gdf[gdf['BUILDING'] == 132]

union = bldg.unary_union
print(f"Union type: {type(union)}")

if hasattr(union, 'interiors'):
    print(f"Holes in union: {len(union.interiors)}")
elif hasattr(union, 'geoms'):
    for i, g in enumerate(union.geoms):
        if hasattr(g, 'interiors'):
            print(f"Part {i} holes: {len(g.interiors)}")
        else:
            print(f"Part {i} has no interiors attribute")
else:
    print("No interiors or geoms attribute found.")
