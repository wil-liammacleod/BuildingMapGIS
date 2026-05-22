from pyproj import Transformer
import numpy as np

transformer = Transformer.from_crs("EPSG:2958", "EPSG:4326", always_xy=True)
lon1, lat1 = transformer.transform(587000.0, 4790000.0)
lon2, lat2 = transformer.transform(587000.0, 4791000.0)

# Calculate bearing from (lon1, lat1) to (lon2, lat2)
# Using standard great-circle bearing formula
lat1_rad = np.radians(lat1)
lat2_rad = np.radians(lat2)
dlon_rad = np.radians(lon2 - lon1)

y = np.sin(dlon_rad) * np.cos(lat2_rad)
x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
bearing = np.degrees(np.arctan2(y, x))

print(f"UTM Grid North bearing: {bearing:.6f} degrees")
